import { useState } from 'react';

interface ControlPanelProps {
  connected: boolean;
  onSpeedChange: (multiplier: number) => void;
  onDisruptionModeToggle: () => void;
  disruptionMode: boolean;
  step: number;
  timeOfDay: number;
  dayOfWeek: number;
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export default function ControlPanel({
  connected, onSpeedChange, onDisruptionModeToggle,
  disruptionMode, step, timeOfDay, dayOfWeek,
}: ControlPanelProps) {
  const [speed, setSpeed] = useState(1);

  const handleSpeed = (m: number) => {
    setSpeed(m);
    onSpeedChange(m);
  };

  const hours = Math.floor(timeOfDay);
  const minutes = Math.round((timeOfDay - hours) * 60);
  const timeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;

  return (
    <>
      <div className="card">
        <div className="card-header">
          <span className="card-title">Simulation Clock</span>
          <span className={`status-dot ${connected ? 'live' : 'danger'}`} />
        </div>
        <div className="card-body">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
            <span style={{ fontSize: 36, fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>
              {timeStr}
            </span>
            <span style={{ fontSize: 14, color: 'var(--text-secondary)', fontWeight: 600 }}>
              {DAYS[dayOfWeek]}
            </span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Step: {step.toLocaleString()}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Simulation Speed</span>
        </div>
        <div className="card-body">
          <div className="speed-selector">
            {[1, 5, 10, 30, 60].map(m => (
              <button key={m} className={`speed-btn ${speed === m ? 'active' : ''}`}
                onClick={() => handleSpeed(m)}>
                {m}x
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Disruption Control</span>
        </div>
        <div className="card-body">
          <button
            className={`btn ${disruptionMode ? 'btn-danger' : 'btn-primary'}`}
            onClick={onDisruptionModeToggle}
            style={{ width: '100%' }}
          >
            {disruptionMode ? '✕ Cancel' : '⚡ Inject Disruption'}
          </button>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.5 }}>
            {disruptionMode
              ? 'Click any node on the map to create a disruption event.'
              : 'Simulate accidents, weather events, or road closures.'}
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">System Info</span>
        </div>
        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div className="tooltip-row">
            <span className="label">Model</span>
            <span className="value">ASTGCN</span>
          </div>
          <div className="tooltip-row">
            <span className="label">Lookback</span>
            <span className="value">12 steps (60 min)</span>
          </div>
          <div className="tooltip-row">
            <span className="label">Horizon</span>
            <span className="value">12 steps (60 min)</span>
          </div>
          <div className="tooltip-row">
            <span className="label">Graph</span>
            <span className="value">15×15 grid</span>
          </div>
          <div className="tooltip-row">
            <span className="label">Vehicles</span>
            <span className="value">30</span>
          </div>
        </div>
      </div>
    </>
  );
}
