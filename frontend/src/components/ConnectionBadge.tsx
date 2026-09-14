interface Props {
  connected: boolean;
}

export function ConnectionBadge({ connected }: Props) {
  return (
    <span className={`connection-badge ${connected ? "connected" : "disconnected"}`}>
      <span className="dot" />
      {connected ? "Connected" : "Reconnecting…"}
    </span>
  );
}
