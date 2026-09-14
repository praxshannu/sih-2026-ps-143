import { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';
import type { TimelineEvent } from '@/lib/api';
import { COLORS } from '@/lib/constants';

interface ForensicBarProps {
  events: TimelineEvent[];
  width?: number;
  height?: number;
}

const SEVERITY_COLORS: Record<string, string> = {
  low: COLORS.success,
  medium: COLORS.warning,
  high: '#F97316',
  critical: COLORS.danger,
};

const EVENT_ICONS: Record<string, string> = {
  SPILL_DETECTED: '!',
  AIS_GAP: '>>',
  VESSEL_SIGHTING: '^',
  SAR_PASS: 'S',
  WEATHER_CHANGE: '~',
  SUSPECT_FLAGGED: '*',
};

export default function ForensicBar({
  events,
  width = 800,
  height = 120,
}: ForensicBarProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const timeExtent = useMemo(() => {
    if (events.length === 0) return [new Date(), new Date()] as [Date, Date];
    const times = events.map((e) => new Date(e.timestamp));
    return d3.extent(times) as [Date, Date];
  }, [events]);

  useEffect(() => {
    if (!svgRef.current || events.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 30, left: 20 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleTime().domain(timeExtent).range([0, innerW]);

    const barHeight = 4;
    const barY = innerH / 2 - barHeight / 2;

    g.append('rect')
      .attr('x', 0)
      .attr('y', barY)
      .attr('width', innerW)
      .attr('height', barHeight)
      .attr('rx', 2)
      .attr('fill', COLORS.border);

    events.forEach((event) => {
      const xPos = x(new Date(event.timestamp));
      const color = SEVERITY_COLORS[event.severity] ?? COLORS.muted;

      g.append('line')
        .attr('x1', xPos)
        .attr('x2', xPos)
        .attr('y1', barY - 10)
        .attr('y2', barY + barHeight + 10)
        .attr('stroke', color)
        .attr('stroke-width', event.severity === 'critical' ? 2 : 1)
        .attr('stroke-dasharray', event.severity === 'low' ? '2,2' : 'none');

      g.append('circle')
        .attr('cx', xPos)
        .attr('cy', innerH / 2)
        .attr('r', event.severity === 'critical' ? 6 : 4)
        .attr('fill', color)
        .attr('stroke', COLORS.bg)
        .attr('stroke-width', 1.5);

      const label = EVENT_ICONS[event.type] ?? '?';
      g.append('text')
        .attr('x', xPos)
        .attr('y', barY - 16)
        .attr('text-anchor', 'middle')
        .attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono')
        .attr('fill', color)
        .text(label);
    });

    const xAxis = d3.axisBottom(x).ticks(8).tickFormat(d3.timeFormat('%H:%M') as (d: d3.NumberValue) => string);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(xAxis)
      .selectAll('text')
      .attr('fill', COLORS.muted)
      .attr('font-size', '9px')
      .attr('font-family', 'JetBrains Mono');

    g.selectAll('.domain').attr('stroke', COLORS.border);
    g.selectAll('.tick line').attr('stroke', COLORS.border);
  }, [events, timeExtent, width, height]);

  return (
    <div className="glass-panel overflow-hidden">
      <div className="border-b border-sentinel-border px-4 py-2">
        <span className="font-mono text-[10px] text-sentinel-muted uppercase">
          Forensic Timeline | D3 Rendered
        </span>
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full"
        viewBox={`0 0 ${width} ${height}`}
      />
    </div>
  );
}
