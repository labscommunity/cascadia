// Compact label for a node's accelerator class. Coloring scales with
// "compute density" without leaning into red/green-as-status: CPU is
// neutral, GPU + NPU get the brand teal because that's where the
// throughput lives.

const STYLES: Record<string, string> = {
  CPU: "bg-paper-2 text-ink-dim border-rule",
  GPU: "bg-mint-wash text-pine border-celadon",
  NPU: "bg-mint-wash text-pine border-celadon",
  iGPU: "bg-mint-wash text-pine border-celadon",
};

export function DeviceChip({ device }: { device: string }) {
  const style = STYLES[device] ?? STYLES["CPU"];
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-sm border font-mono text-[11px] tracking-wider uppercase ${style}`}
    >
      {device}
    </span>
  );
}
