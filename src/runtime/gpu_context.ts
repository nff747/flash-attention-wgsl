/**
 * flash-attention-wgsl: WebGPU Device & Adapter Context Manager
 * Device acquisition, limit inspection, and capability detection.
 */

export interface GPUContextInfo {
  readonly device: GPUDevice;
  readonly adapter: GPUAdapter;
  readonly maxWorkgroupSizeX: number;
  readonly maxComputeInvocations: number;
  readonly maxStorageBufferSize: number;
  readonly hasSubgroups: boolean;
  readonly hasF16: boolean;
}

export async function initWebGPUContext(
  requiredFeatures: GPUFeatureName[] = []
): Promise<GPUContextInfo | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance'
  });

  if (!adapter) {
    return null;
  }

  const availableFeatures: GPUFeatureName[] = [];
  const hasSubgroups = adapter.features.has('subgroups' as GPUFeatureName);
  const hasF16 = adapter.features.has('shader-f16');

  for (const feat of requiredFeatures) {
    if (adapter.features.has(feat)) {
      availableFeatures.push(feat);
    }
  }

  const device = await adapter.requestDevice({
    requiredFeatures: availableFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
    }
  });

  return {
    device,
    adapter,
    maxWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxComputeInvocations: device.limits.maxComputeInvocationsPerWorkgroup,
    maxStorageBufferSize: device.limits.maxStorageBufferBindingSize,
    hasSubgroups,
    hasF16
  };
}
