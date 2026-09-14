import type { StatusMessage } from "../types";

interface Props {
  status: StatusMessage | null;
}

const MODE_EXPLANATION: Record<StatusMessage["mode"], string> = {
  predicted: "The arm is following instructions from the policy.",
  fallback:
    "No fresh instruction arrived in time, so the arm is safely holding its last position.",
};

export function TelemetryPanel({ status }: Props) {
  return (
    <section className="panel">
      <h2>Loop telemetry</h2>
      {status === null ? (
        <p className="muted">Waiting for status…</p>
      ) : (
        <>
          <dl className="telemetry-grid">
            <dt className="hint" title="How many times per second the simulation is currently updating. Should sit near 60.">
              Control rate
            </dt>
            <dd>{status.paused ? "paused" : `${status.control_hz.toFixed(1)} Hz`}</dd>

            <dt className="hint" title="What's currently deciding the arm's motion -- see the explanation below.">
              Action source
            </dt>
            <dd>
              <span className={`mode-badge ${status.mode}`}>{status.mode}</span>
            </dd>

            <dt className="hint" title="How old the instruction currently driving the arm is.">
              Action age
            </dt>
            <dd>{status.action_age_ms === null ? "—" : `${status.action_age_ms.toFixed(1)} ms`}</dd>

            <dt className="hint" title="How many times the simulation has had to fall back to 'just hold position' since it started. Not an error count.">
              Fallback steps
            </dt>
            <dd>{status.fallback_steps.toLocaleString()}</dd>

            <dt>Frame</dt>
            <dd>#{status.frame_id}</dd>

            <dt className="hint" title="Whether small random variations (lighting, camera position, object weight, friction) are applied each time the episode resets.">
              Domain randomization
            </dt>
            <dd>{status.domain_randomization_enabled ? "on" : "off"}</dd>
          </dl>
          <p className="mode-explanation">{MODE_EXPLANATION[status.mode]}</p>
        </>
      )}
    </section>
  );
}
