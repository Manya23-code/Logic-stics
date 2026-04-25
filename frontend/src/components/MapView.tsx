import { useRef, useEffect, useState, useCallback } from 'react';
import type { GraphData, SimState } from '../hooks/useSimulation';

interface MapViewProps {
  graph: GraphData | null;
  state: SimState | null;
  disruptionMode: boolean;
  onNodeClick: (nodeId: number) => void;
}

function speedToColor(speed: number): string {
  if (speed > 60) return '#34d399';
  if (speed > 40) return '#fbbf24';
  if (speed > 20) return '#ef4444';
  return '#991b1b';
}

function speedToWidth(speed: number): number {
  if (speed > 60) return 2;
  if (speed > 40) return 2.5;
  if (speed > 20) return 3;
  return 4;
}

export default function MapView({ graph, state, disruptionMode, onNodeClick }: MapViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; nodeId: number; speed: number } | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Handle resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Coordinate mapping
  const getNodePos = useCallback((nodeId: number) => {
    if (!graph) return { x: 0, y: 0 };
    const node = graph.nodes[nodeId];
    if (!node) return { x: 0, y: 0 };
    const padding = 60;
    const w = dimensions.width - padding * 2;
    const h = dimensions.height - padding * 2;
    const gs = graph.grid_size;
    return {
      x: padding + (node.x / (gs - 1)) * w,
      y: padding + (node.y / (gs - 1)) * h,
    };
  }, [graph, dimensions]);

  // Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !graph || dimensions.width === 0) return;

    canvas.width = dimensions.width * 2;
    canvas.height = dimensions.height * 2;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(2, 2);

    // Background
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    // Grid subtle background
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i < dimensions.width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, dimensions.height);
      ctx.stroke();
    }
    for (let j = 0; j < dimensions.height; j += 40) {
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(dimensions.width, j);
      ctx.stroke();
    }

    const speeds = state?.traffic?.speeds || [];

    // Draw edges (road segments)
    for (const edge of graph.edges) {
      const from = getNodePos(edge.source);
      const to = getNodePos(edge.target);
      const targetSpeed = speeds[edge.target] || 50;
      const color = speedToColor(targetSpeed);
      const width = speedToWidth(targetSpeed);

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw bottleneck halos
    const bottlenecks = state?.bottleneck_nodes || [];
    for (const bn of bottlenecks) {
      const pos = getNodePos(bn);
      const severity = state?.prediction?.bottleneck_severity?.[String(bn)] || 0.5;
      const radius = 12 + severity * 15;
      const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius);
      gradient.addColorStop(0, `rgba(239, 68, 68, ${0.4 * severity})`);
      gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw disruption markers
    for (const d of (state?.traffic?.disruptions || [])) {
      const pos = getNodePos(d.node_id);
      // Pulsing ring
      const pulseR = 18 + Math.sin(Date.now() / 300) * 5;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(239, 68, 68, 0.6)`;
      ctx.lineWidth = 2;
      ctx.stroke();
      // Icon
      ctx.fillStyle = '#ef4444';
      ctx.font = '14px Inter';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('⚠', pos.x, pos.y);
    }

    // Draw nodes
    for (const node of graph.nodes) {
      const pos = getNodePos(node.id);
      const speed = speeds[node.id] || 50;
      const isBottleneck = bottlenecks.includes(node.id);

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isBottleneck ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = isBottleneck ? '#ef4444' : speedToColor(speed);
      ctx.fill();

      if (isBottleneck) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 7, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // Draw vehicles
    const vehicles = state?.fleet?.vehicles || [];
    for (const v of vehicles) {
      if (v.status === 'delivered') continue;
      const pos = getNodePos(v.current_node);
      // Vehicle dot
      const color = v.status === 'rerouted' ? '#00d4ff' : '#a855f7';
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      // Glow
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = `${color}66`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

  }, [graph, state, dimensions, getNodePos]);

  // Mouse interaction
  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    if (!graph) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let closestNode = -1;
    let closestDist = Infinity;

    for (const node of graph.nodes) {
      const pos = getNodePos(node.id);
      const d = Math.sqrt((mx - pos.x) ** 2 + (my - pos.y) ** 2);
      if (d < closestDist && d < 20) {
        closestDist = d;
        closestNode = node.id;
      }
    }

    if (closestNode >= 0) {
      onNodeClick(closestNode);
    }
  }, [graph, getNodePos, onNodeClick]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!graph || !state) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    for (const node of graph.nodes) {
      const pos = getNodePos(node.id);
      const d = Math.sqrt((mx - pos.x) ** 2 + (my - pos.y) ** 2);
      if (d < 12) {
        setTooltip({ x: e.clientX, y: e.clientY, nodeId: node.id, speed: state.traffic?.speeds?.[node.id] || 0 });
        return;
      }
    }
    setTooltip(null);
  }, [graph, state, getNodePos]);

  return (
    <div className="map-container" ref={containerRef}>
      <canvas
        ref={canvasRef}
        className="map-canvas"
        style={{ width: '100%', height: '100%' }}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />

      {disruptionMode && (
        <div className="disruption-mode">
          <span>🎯</span> Click on the map to inject a disruption
        </div>
      )}

      <div className="map-legend">
        <div className="legend-title">Traffic Speed</div>
        <div className="legend-bar" />
        <div className="legend-labels">
          <span>0</span>
          <span>20</span>
          <span>40</span>
          <span>60+</span>
        </div>
        <div style={{ marginTop: 8, fontSize: 10, color: 'var(--text-muted)' }}>
          <span style={{ color: '#a855f7' }}>●</span> Vehicle&nbsp;
          <span style={{ color: '#00d4ff' }}>●</span> Rerouted&nbsp;
          <span style={{ color: '#ef4444' }}>●</span> Bottleneck
        </div>
      </div>

      {tooltip && (
        <div className="node-tooltip" style={{ left: tooltip.x + 15, top: tooltip.y - 60, position: 'fixed' }}>
          <div className="tooltip-title">Node #{tooltip.nodeId}</div>
          <div className="tooltip-row">
            <span className="label">Speed</span>
            <span className="value" style={{ color: speedToColor(tooltip.speed) }}>
              {tooltip.speed.toFixed(1)} km/h
            </span>
          </div>
          <div className="tooltip-row">
            <span className="label">Status</span>
            <span className="value">
              {state?.bottleneck_nodes?.includes(tooltip.nodeId) ? '⚠️ Bottleneck' : '✅ Normal'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
