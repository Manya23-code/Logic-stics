import { useState, useCallback, useMemo } from 'react';
import { useSimulation } from './hooks/useSimulation';
import MapView from './components/MapView';
import ControlPanel from './components/ControlPanel';
import AlertFeed from './components/AlertFeed';
import KPIDashboard from './components/KPIDashboard';
import './index.css';

export default function App() {
  // useSimulation hook returns state, graph, connected, etc.
  const { state, graph, connected, events, injectDisruption, setSpeed } = useSimulation();
  const [disruptionMode, setDisruptionMode] = useState(false);

  // Casting state to any to avoid TypeScript property missing errors during build
  const s = state as any;

  // Time Travel Logic: Passing offset hours to the backend
  const handleTimeTravel = (hours: number) => {
    if (setSpeed) {
      setSpeed(hours);
    }
  };

  const handleNodeClick = useCallback((nodeId: number) => {
    if (disruptionMode && injectDisruption) {
      console.log(`Injecting disruption at node: ${nodeId}`);
      // 0.05 severity = High impact (Road closure)
      injectDisruption(nodeId, 0.05, 'accident');  
      setDisruptionMode(false);
    }
  }, [disruptionMode, injectDisruption]);

  // Safe KPI calculations
  const onTimeRate = useMemo(() => {
    const total = (s?.fleet?.total_deliveries || 0) + (s?.fleet?.total_reroutes || 0);
    if (total === 0) return 95;
    return Math.min(99, ((s?.fleet?.total_deliveries || 0) / total) * 100);
  }, [s?.fleet]);

  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="logo">
          <div className="logo-icon">LG</div>
          <div>
            <div className="logo-text">Logic-stics</div>
            <div className="logo-subtitle">Predictive Digital Twin</div>
          </div>
        </div>
        
        <div className="header-status">
          <div className="status-badge">
             Live DU Speed: {s?.live_anchor_speed ? Math.round(s.live_anchor_speed) : '--'} km/h
          </div>
          
          <div className={`status-badge ${s?.is_live_synced === false ? 'simulating-glow' : ''}`}>
            <span className={`status-dot ${connected ? (s?.is_live_synced !== false ? 'live' : 'warning') : 'danger'}`} />
            {connected 
              ? (s?.is_live_synced !== false ? 'Cloud Sync: Active' : `Time Travel: ${s?.time_offset || 0}h Offset`) 
              : 'Connecting to Render Server…'}
          </div>

          <div className="status-badge">ASTGCN ML Active</div>
          <div className="status-badge">{s?.bottleneck_nodes?.length || 0} Bottlenecks Predicted</div>
        </div>
      </header>

      {/* ── Left Sidebar: Controls ── */}
      <aside className="sidebar-left">
        <ControlPanel
          connected={connected}
          onSpeedChange={handleTimeTravel} 
          onDisruptionModeToggle={() => setDisruptionMode(m => !m)}
          disruptionMode={disruptionMode}
          step={s?.step || 0}
          timeOfDay={s?.traffic?.time_of_day ?? 0}
          dayOfWeek={s?.traffic?.day_of_week ?? 0}
        />
        <div style={{padding: '12px', fontSize: '11px', color: '#64748b', borderTop: '1px solid #1e293b', marginTop: '10px'}}>
           <strong>Time Sync:</strong> 1x = Live (IST) <br/>
           <strong>Travel:</strong> Use 5x for -5hrs ago
        </div>
      </aside>

      {/* ── Main Map: Matrix ── */}
      <main className="main-map">
        {/* We check if graph exists before rendering to avoid empty space */}
        {graph ? (
          <MapView
            graph={graph}
            state={state}
            disruptionMode={disruptionMode}
            onNodeClick={handleNodeClick}
          />
        ) : (
          <div className="loading-map">Initializing Digital Twin Grid...</div>
        )}
      </main>

      {/* ── Right Sidebar: Alerts ── */}
      <aside className="sidebar-right">
        <AlertFeed
          events={events}
          bottleneckCount={s?.bottleneck_nodes?.length || 0}
          meanSpeed={s?.live_anchor_speed || 50}
        />
      </aside>

      {/* ── Bottom: KPIs ── */}
      <div className="bottom-panel">
        <KPIDashboard
          activeShipments={s?.fleet?.active_count || 0}
          onTimeRate={onTimeRate}
          totalReroutes={s?.fleet?.total_reroutes || 0}
          totalDeliveries={s?.fleet?.total_deliveries || 0}
          timeSaved={s?.fleet?.total_time_saved || 0}
          avgDeliveryTime={s?.fleet?.avg_delivery_time || 0}
        />
      </div>
    </div>
  );
}