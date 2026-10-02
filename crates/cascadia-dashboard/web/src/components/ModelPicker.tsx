import { useEffect, useState } from "react";

import { api, type ModelEntry } from "@/lib/api";

type Props = {
  value: string;
  onChange: (id: string) => void;
};

/**
 * Coordinator-side model selector. Cascadia loads a single model per
 * worker today, so the list usually has one entry — we still render a
 * select for clarity (and to leave room for multi-model coordinators
 * later).
 */
export function ModelPicker({ value, onChange }: Props) {
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .models()
      .then((r) => {
        if (!alive) return;
        setModels(r.data);
        if (!value && r.data.length > 0) onChange(r.data[0].id);
      })
      .catch((e) => {
        if (alive) setError((e as Error).message);
      });
    return () => {
      alive = false;
    };
    // Mount-only on purpose: the model list rarely changes mid-session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return <span className="label-mono text-state-error">/v1/models {error}</span>;
  }
  if (!models) {
    return <span className="label-mono">loading…</span>;
  }
  if (models.length === 0) {
    return <span className="label-mono">no models loaded</span>;
  }
  // Always render the <select> — even with a single model. The dropdown
  // affordance signals "this is a control you can interact with" and
  // future-proofs against multi-model coordinators without a code path
  // change. With one entry the user just sees the model name plus a
  // disabled-looking chevron, which is fine.
  return (
    <label className="flex items-center gap-2">
      <span className="label-mono">Model</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="font-mono text-[13px] text-ink bg-paper border border-rule rounded-sm px-2 py-1 focus:outline-none focus:border-persian hover:border-pine transition-colors"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.id}
          </option>
        ))}
      </select>
    </label>
  );
}
