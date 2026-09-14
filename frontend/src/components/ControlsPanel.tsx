import { useState } from "react";
import { pauseSim, resetEpisode, resumeSim, setDomainRandomization, stepSim } from "../api";
import type { StatusMessage } from "../types";

interface Props {
  status: StatusMessage | null;
}

export function ControlsPanel({ status }: Props) {
  const [error, setError] = useState<string | null>(null);

  async function run(action: () => Promise<void>) {
    try {
      setError(null);
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const paused = status?.paused ?? false;

  return (
    <section className="panel">
      <h2>Controls</h2>
      <div className="controls-row">
        <button
          title="Starts a new episode: the arm and everything else resets to a fresh starting position."
          onClick={() => run(resetEpisode)}
        >
          Reset episode
        </button>
        <button
          title={
            paused
              ? "Resume: let physics keep running."
              : "Pause: freeze physics completely. The arm and camera stop updating."
          }
          onClick={() => run(paused ? resumeSim : pauseSim)}
        >
          {paused ? "Resume" : "Pause"}
        </button>
        <button
          title="Advance exactly one instant while paused, for inspecting things one step at a time."
          onClick={() => run(stepSim)}
          disabled={!paused}
        >
          Step
        </button>
      </div>
      <label className="dr-toggle hint" title="When on, small random variations (lighting, camera position, object weight, friction) are applied each time the episode resets.">
        <input
          type="checkbox"
          checked={status?.domain_randomization_enabled ?? false}
          onChange={(e) => run(() => setDomainRandomization(e.target.checked))}
        />
        Domain randomization
      </label>
      {error !== null && <p className="error">{error}</p>}
    </section>
  );
}
