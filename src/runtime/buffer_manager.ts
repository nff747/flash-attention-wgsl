/**
 * flash-attention-wgsl: GPU Buffer Allocator & Staging Manager
 * Aligned memory transfers between CPU Float32Array and GPU storage buffers.
 */

export class GPUBufferManager {
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Creates a GPU storage buffer populated with Float32Array data.
   */
  createStorageBufferFromData(
    data: Float32Array,
    label = 'StorageBuffer'
  ): GPUBuffer {
    const byteLength = Math.max(16, data.byteLength);
    // Align to 4 bytes
    const alignedSize = Math.ceil(byteLength / 4) * 4;

    const buffer = this.device.createBuffer({
      label,
      size: alignedSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC
    });

    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  }

  /**
   * Creates an empty GPU storage buffer of specified float count.
   */
  createEmptyStorageBuffer(
    elementCount: number,
    label = 'EmptyStorageBuffer'
  ): GPUBuffer {
    const size = Math.max(16, Math.ceil((elementCount * 4) / 4) * 4);
    return this.device.createBuffer({
      label,
      size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC
    });
  }

  /**
   * Reads data from a GPU buffer back to CPU Float32Array via a mapped staging buffer.
   */
  async readBufferToCPU(
    gpuBuffer: GPUBuffer,
    byteLength: number
  ): Promise<Float32Array> {
    const alignedSize = Math.ceil(byteLength / 4) * 4;

    const stagingBuffer = this.device.createBuffer({
      label: 'ReadStagingBuffer',
      size: alignedSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, byteLength);
    this.device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = stagingBuffer.getMappedRange(0, byteLength);
    const result = new Float32Array(copyArrayBuffer.slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return result;
  }
}
