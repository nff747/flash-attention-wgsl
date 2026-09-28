/**
 * flash-attention-wgsl: WebGPU FlashAttention Execution Pipeline
 * Builds compute pipelines, bind groups, and orchestrates grid dispatches.
 */

import {
  compileFlashAttentionV2WGSL,
  compileFlashAttentionVec4WGSL
} from '../shaders';
import { GPUBufferManager } from './buffer_manager';
import { Tensor4D, AttentionConfig } from '../math/types';
import { defaultScale } from '../math/numerical_stability';

export interface PipelineExecutionOptions {
  useVec4?: boolean;
}

export class FlashAttentionPipeline {
  private device: GPUDevice;
  private bufferManager: GPUBufferManager;

  constructor(device: GPUDevice) {
    this.device = device;
    this.bufferManager = new GPUBufferManager(device);
  }

  /**
   * Executes FlashAttention compute shader on the GPU.
   */
  async execute(
    Q: Tensor4D,
    K: Tensor4D,
    V: Tensor4D,
    config: AttentionConfig,
    options: PipelineExecutionOptions = {}
  ): Promise<Float32Array> {
    const { batch, heads, seqLen, headDim } = Q.shape;
    const blockSizeR = config.blockSizeR ?? 32;
    const blockSizeC = config.blockSizeC ?? 32;
    const scale = config.scale ?? defaultScale(headDim);
    const isCausal = config.causal ? 1 : 0;

    const useVec4 = options.useVec4 ?? (headDim % 4 === 0);

    // 1. Generate specialized WGSL shader code
    const wgslSource = useVec4
      ? compileFlashAttentionVec4WGSL({ blockSizeR, blockSizeC, headDim })
      : compileFlashAttentionV2WGSL({ blockSizeR, blockSizeC, headDim });

    const shaderModule = this.device.createShaderModule({
      label: useVec4 ? 'FlashAttentionVec4Shader' : 'FlashAttentionV2Shader',
      code: wgslSource
    });

    // 2. Uniform buffer creation
    const uniformArray = new ArrayBuffer(32);
    const uniformUint = new Uint32Array(uniformArray);
    const uniformFloat = new Float32Array(uniformArray);

    uniformUint[0] = batch;
    uniformUint[1] = heads;
    uniformUint[2] = seqLen;
    uniformUint[3] = headDim;
    uniformFloat[4] = scale;
    uniformUint[5] = isCausal;
    uniformUint[6] = blockSizeR;
    uniformUint[7] = blockSizeC;

    const uniformBuffer = this.device.createBuffer({
      label: 'AttentionUniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // 3. Upload Q, K, V to GPU
    const qBuffer = this.bufferManager.createStorageBufferFromData(Q.data, 'Buffer_Q');
    const kBuffer = this.bufferManager.createStorageBufferFromData(K.data, 'Buffer_K');
    const vBuffer = this.bufferManager.createStorageBufferFromData(V.data, 'Buffer_V');

    const totalOutputFloats = batch * heads * seqLen * headDim;
    const oBuffer = this.bufferManager.createEmptyStorageBuffer(totalOutputFloats, 'Buffer_O');

    // 4. Create Compute Pipeline
    const pipeline = this.device.createComputePipeline({
      label: 'FlashAttentionComputePipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });

    // 5. Create Bind Group
    const bindGroup = this.device.createBindGroup({
      label: 'FlashAttentionBindGroup',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: qBuffer } },
        { binding: 2, resource: { buffer: kBuffer } },
        { binding: 3, resource: { buffer: vBuffer } },
        { binding: 4, resource: { buffer: oBuffer } }
      ]
    });

    // 6. Encode and Dispatch Compute Pass
    const Tr = Math.ceil(seqLen / blockSizeR);
    const commandEncoder = this.device.createCommandEncoder({ label: 'FlashAttentionCommandEncoder' });
    const passEncoder = commandEncoder.beginComputePass({ label: 'FlashAttentionComputePass' });

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Tr, heads, batch);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // 7. Read Output Back to CPU
    const outputData = await this.bufferManager.readBufferToCPU(
      oBuffer,
      totalOutputFloats * 4
    );

    // Cleanup GPU resources
    uniformBuffer.destroy();
    qBuffer.destroy();
    kBuffer.destroy();
    vBuffer.destroy();
    oBuffer.destroy();

    return outputData;
  }
}
