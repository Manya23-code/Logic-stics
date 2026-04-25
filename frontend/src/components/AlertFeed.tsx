import { motion, AnimatePresence } from 'framer-motion';
import type { SimEvent } from '../hooks/useSimulation';

interface AlertFeedProps {
  events: SimEvent[];
  bottleneckCount: number;
  meanSpeed: number;
}

function eventIcon(type: string) {
  switch (type) {
    case 'reroute': return '🔄';
    case 'disruption_injected': return '⚠️';
    case 'delivery': return '📦';
    default: return '📡';
  }
}

function eventClass(type: string) {
  switch (type) {
    case 'reroute': return 'reroute';
    case 'disruption_injected': return 'disruption';
    case 'delivery': return 'delivery';
    default: return 'bottleneck';
  }
}

function eventTitle(e: SimEvent) {
  switch (e.type) {
    case 'reroute':
      return `Vehicle #${e.vehicle_id} Rerouted`;
    case 'disruption_injected':
      return `${e.event_type?.charAt(0).toUpperCase()}${e.event_type?.slice(1)} at Node #${e.node_id}`;
    case 'delivery':
      return `Vehicle #${e.vehicle_id} Delivered`;
    default:
      return 'Event';
  }
}

function eventBody(e: SimEvent) {
  switch (e.type) {
    case 'reroute':
      return `Saved ${e.time_saved?.toFixed(0)}s via dynamic rerouting`;
    case 'disruption_injected':
      return `Severity: ${((1 - (e.severity || 0)) * 100).toFixed(0)}% impact`;
    case 'delivery':
      return `Completed in ${e.total_time?.toFixed(0)}s`;
    default:
      return '';
  }
}

export default function AlertFeed({ events, bottleneckCount, meanSpeed }: AlertFeedProps) {
  return (
    <>
      <div className="section-label">Live Feed</div>

      {/* Summary cards */}
      <div style={{ display: 'flex', gap: 8 }}>
        <div className="card" style={{ flex: 1, padding: 12 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Bottlenecks
          </div>
          <div style={{
            fontSize: 24, fontWeight: 800, fontFamily: 'var(--font-mono)',
            color: bottleneckCount > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)',
          }}>
            {bottleneckCount}
          </div>
        </div>
        <div className="card" style={{ flex: 1, padding: 12 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Avg Speed
          </div>
          <div style={{
            fontSize: 24, fontWeight: 800, fontFamily: 'var(--font-mono)',
            color: meanSpeed > 40 ? 'var(--accent-emerald)' : meanSpeed > 20 ? 'var(--accent-amber)' : 'var(--accent-red)',
          }}>
            {meanSpeed.toFixed(0)}
          </div>
        </div>
      </div>

      <div className="section-label" style={{ marginTop: 4 }}>Events</div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        <AnimatePresence initial={false}>
          {events.slice(0, 30).map((event, i) => (
            <motion.div
              key={`${event.type}-${i}-${event.vehicle_id || event.node_id}`}
              initial={{ opacity: 0, x: 30, height: 0 }}
              animate={{ opacity: 1, x: 0, height: 'auto' }}
              exit={{ opacity: 0, x: -30, height: 0 }}
              transition={{ duration: 0.3 }}
              style={{ marginBottom: 8 }}
            >
              <div className={`alert-card ${eventClass(event.type)}`}>
                <div className="alert-time">
                  {eventIcon(event.type)} Step {event.step || '—'}
                </div>
                <div className="alert-title">{eventTitle(event)}</div>
                <div className="alert-body">{eventBody(event)}</div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {events.length === 0 && (
          <div style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)', fontSize: 13 }}>
            Waiting for events…
          </div>
        )}
      </div>
    </>
  );
}
