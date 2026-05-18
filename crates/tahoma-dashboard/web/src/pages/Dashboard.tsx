export function Dashboard() {
  return (
    <div className="max-w-container mx-auto px-6 py-12 space-y-12">
      <header className="space-y-3 max-w-3xl">
        <div className="label-mono">Cluster overview</div>
        <h1 className="font-display text-display text-ink">Nodes &amp; latencies</h1>
        <p className="text-ink-dim text-[17px] leading-relaxed">
          A live view of every Intel device this Tahoma coordinator has seen.
          Latency and bandwidth come from measured probes between peers — not
          declared specs — so placement decisions can lean on what the
          network actually does.
        </p>
      </header>

      <section className="surface p-8">
        <div className="label-mono mb-4">Nodes</div>
        <div className="font-mono text-sm text-ink-low">
          node grid renders in the next commit
        </div>
      </section>

      <section className="surface p-8">
        <div className="label-mono mb-4">Latency matrix</div>
        <div className="font-mono text-sm text-ink-low">
          measured edge matrix renders in the next commit
        </div>
      </section>
    </div>
  );
}
