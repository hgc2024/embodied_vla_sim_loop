import type { FrameMessage } from "../types";

interface Props {
  frame: FrameMessage | null;
}

// Fixed display ranges purely for the bar visualization below -- not a claim
// about the real joint limits (see sim/assets/placeholder_arm.xml for those).
const POSITION_RANGE_RAD = Math.PI;
const VELOCITY_RANGE_RAD_S = 5;

function Bar({ value, range }: { value: number; range: number }) {
  const clamped = Math.max(-range, Math.min(range, value));
  const percentFromCenter = (clamped / range) * 50; // -50..50
  return (
    <div className="bar-track">
      <div
        className="bar-fill"
        style={{
          left: percentFromCenter < 0 ? `${50 + percentFromCenter}%` : "50%",
          width: `${Math.abs(percentFromCenter)}%`,
        }}
      />
      <div className="bar-center" />
    </div>
  );
}

export function JointStatePanel({ frame }: Props) {
  return (
    <section className="panel">
      <h2>Joint state</h2>
      <p className="muted small">
        The arm's own sense of each joint's angle and speed -- like a real robot's
        internal sensors.
      </p>
      {frame === null ? (
        <p className="muted">Waiting for the first observation…</p>
      ) : (
        <table className="joint-table">
          <thead>
            <tr>
              <th>Joint</th>
              <th>Position (rad)</th>
              <th>Velocity (rad/s)</th>
            </tr>
          </thead>
          <tbody>
            {frame.joint_positions.map((position, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>
                  <Bar value={position} range={POSITION_RANGE_RAD} />
                  <span className="bar-value">{position.toFixed(2)}</span>
                </td>
                <td>
                  <Bar value={frame.joint_velocities[i] ?? 0} range={VELOCITY_RANGE_RAD_S} />
                  <span className="bar-value">{(frame.joint_velocities[i] ?? 0).toFixed(2)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
