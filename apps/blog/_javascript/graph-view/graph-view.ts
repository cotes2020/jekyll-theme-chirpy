import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm';

export interface GraphPalette {
  link: string;
  linkDim: string;
  linkHi: string;
  text: string;
  nodeStroke: string;
  nodeFillDefault: string;
}

export interface GraphNodeInput {
  id: string;
  label?: string;
  group?: string;
  href?: string;
}

export interface GraphLinkInput {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNodeInput[];
  links?: GraphLinkInput[];
}

export interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label?: string;
  group?: string;
  href?: string;
}

export interface CreateGraphViewOptions {
  container: HTMLElement | string;
  data?: GraphData;
  dataUrl?: string;
  fetchOptions?: RequestInit;
  onNodeOpen?: (node: GraphNode) => void;
  getPalette?: (root: HTMLElement) => GraphPalette;
}

function defaultPalette(): GraphPalette {
  const mode = document.documentElement.getAttribute('data-mode');
  const dark =
    mode === 'dark' ||
    (mode !== 'light' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
  return {
    link: dark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.16)',
    linkDim: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
    linkHi: dark ? 'rgba(147,197,253,0.95)' : 'rgba(37,99,235,0.85)',
    text: dark ? '#e5e7eb' : '#1f2937',
    nodeStroke: dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.2)',
    nodeFillDefault: dark ? '#6366f1' : '#4f46e5'
  };
}

export async function createGraphView(
  options: CreateGraphViewOptions
): Promise<{ destroy: () => void }> {
  const {
    container,
    data: initialData,
    dataUrl,
    fetchOptions,
    onNodeOpen,
    getPalette = defaultPalette
  } = options;

  const root =
    typeof container === 'string'
      ? document.querySelector<HTMLElement>(container)
      : container;
  if (!root) throw new Error('createGraphView: container not found');
  const elRoot: HTMLElement = root;

  const width = () => Math.max(elRoot.clientWidth || 640, 320);
  const height = () => Math.max(elRoot.clientHeight || 480, 240);

  const svg = d3
    .select(elRoot)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .style('display', 'block')
    .style('touch-action', 'none');

  const gRoot = svg.append('g');
  const gLinks = gRoot.append('g').attr('class', 'graph-links');
  const gNodes = gRoot.append('g').attr('class', 'graph-nodes');

  const zoom = d3
    .zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.15, 5])
    .on('zoom', (ev) => {
      gRoot.attr('transform', ev.transform);
    });
  svg.call(zoom);

  let data: GraphData | undefined = initialData;
  if (dataUrl && !data) {
    const res = await fetch(dataUrl, fetchOptions);
    if (!res.ok) throw new Error(`graph fetch failed: ${res.status}`);
    data = (await res.json()) as GraphData;
  }
  if (!data || !Array.isArray(data.nodes))
    throw new Error('createGraphView: invalid data');

  const nodes: GraphNode[] = data.nodes.map((n) => ({ ...n }));
  const rawLinks = (data.links || []).map((l) => ({ ...l }));
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const links = rawLinks.filter(
    (l) => nodeById.has(l.source) && nodeById.has(l.target)
  ) as d3.SimulationLinkDatum<GraphNode>[];

  const groupKeys = [...new Set(nodes.map((n) => n.group).filter(Boolean))] as string[];
  const colorScale = d3.scaleOrdinal(d3.schemeTableau10).domain(groupKeys);

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      'link',
      d3
        .forceLink<GraphNode, d3.SimulationLinkDatum<GraphNode>>(links)
        .id((d) => d.id)
        .distance(48)
    )
    .force('charge', d3.forceManyBody().strength(-140))
    .force('center', d3.forceCenter(width() / 2, height() / 2))
    .force('collision', d3.forceCollide().radius(32));

  const linkSel = gLinks
    .selectAll<SVGLineElement, d3.SimulationLinkDatum<GraphNode>>('line')
    .data(links, (d) => {
      const s = typeof d.source === 'object' ? d.source.id : String(d.source);
      const t = typeof d.target === 'object' ? d.target.id : String(d.target);
      return `${s}-${t}`;
    })
    .join('line')
    .attr('stroke-width', 1.2);

  const nodeSel = gNodes
    .selectAll<SVGGElement, GraphNode>('g')
    .data(nodes, (d) => d.id)
    .join('g')
    .call(
      d3
        .drag<SVGGElement, GraphNode>()
        .on('start', (ev) => {
          if (!ev.active) simulation.alphaTarget(0.35).restart();
          ev.subject.fx = ev.subject.x;
          ev.subject.fy = ev.subject.y;
        })
        .on('drag', (ev) => {
          ev.subject.fx = ev.x;
          ev.subject.fy = ev.y;
        })
        .on('end', (ev) => {
          if (!ev.active) simulation.alphaTarget(0);
          ev.subject.fx = null;
          ev.subject.fy = null;
        })
    );

  nodeSel.append('circle').attr('r', 10).attr('stroke-width', 1.4);
  nodeSel
    .append('text')
    .attr('dx', 14)
    .attr('dy', 4)
    .attr('font-size', 11)
    .style('pointer-events', 'none')
    .text((d) => d.label || d.id);

  const adj = new Map<string, Set<string>>();
  for (const n of nodes) adj.set(n.id, new Set());
  for (const l of links) {
    const s = typeof l.source === 'object' ? l.source.id : String(l.source);
    const t = typeof l.target === 'object' ? l.target.id : String(l.target);
    adj.get(s)?.add(t);
    adj.get(t)?.add(s);
  }

  let focusId: string | null = null;

  function applyStyles(): void {
    const pal = getPalette(elRoot);
    linkSel.attr('stroke', (d) => {
      if (!focusId) return pal.link;
      const s =
        typeof d.source === 'object' ? d.source.id : String(d.source);
      const t =
        typeof d.target === 'object' ? d.target.id : String(d.target);
      return s === focusId || t === focusId ? pal.linkHi : pal.linkDim;
    });

    nodeSel.selectAll<SVGCircleElement, GraphNode>('circle').each(function (
      d
    ) {
      const el = d3.select(this);
      const pal = getPalette(elRoot);
      const fill =
        d.group != null ? colorScale(d.group) : pal.nodeFillDefault;
      const hi =
        !focusId ||
        focusId === d.id ||
        (focusId ? adj.get(focusId)?.has(d.id) : false);
      el.attr('fill', fill)
        .attr('stroke', pal.nodeStroke)
        .attr('opacity', hi ? 1 : 0.28);
    });

    nodeSel.selectAll<SVGTextElement, GraphNode>('text').each(function (d) {
      const pal = getPalette(elRoot);
      const hi =
        !focusId ||
        focusId === d.id ||
        (focusId ? adj.get(focusId)?.has(d.id) : false);
      d3.select(this).attr('fill', pal.text).attr('opacity', hi ? 1 : 0.35);
    });
  }

  nodeSel
    .on('mouseenter', (_ev, d) => {
      focusId = d.id;
      applyStyles();
    })
    .on('mouseleave', () => {
      focusId = null;
      applyStyles();
    })
    .on('click', (ev, d) => {
      ev.stopPropagation();
      if (typeof onNodeOpen === 'function') onNodeOpen(d);
      else if (d.href) window.location.href = d.href;
    });

  svg.on('click', () => {
    focusId = null;
    applyStyles();
  });

  simulation.on('tick', () => {
    linkSel
      .attr('x1', (d) =>
        typeof d.source === 'object' ? d.source.x ?? 0 : 0
      )
      .attr('y1', (d) =>
        typeof d.source === 'object' ? d.source.y ?? 0 : 0
      )
      .attr('x2', (d) =>
        typeof d.target === 'object' ? d.target.x ?? 0 : 0
      )
      .attr('y2', (d) =>
        typeof d.target === 'object' ? d.target.y ?? 0 : 0
      );
    nodeSel.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
  });

  applyStyles();

  const ro = new ResizeObserver(() => {
    simulation.force('center', d3.forceCenter(width() / 2, height() / 2));
    simulation.alpha(0.25).restart();
  });
  ro.observe(elRoot);

  const mo = new MutationObserver(() => applyStyles());
  mo.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-mode', 'data-theme']
  });

  return {
    destroy() {
      mo.disconnect();
      ro.disconnect();
      simulation.stop();
      svg.remove();
    }
  };
}
