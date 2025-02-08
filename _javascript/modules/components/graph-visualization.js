import { select, zoom, drag, forceSimulation, forceLink, forceManyBody, forceCenter, zoomIdentity } from "d3";

const d3 = {
  select,
  zoom,
  drag,
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  zoomIdentity
};

const nodes = [];
const links = [];
const tags = new Set();  // To track unique tags

// Fetch the JSON graph data
function loadGraphData() {
    fetch("/assets/js/data/graph.json")
    .then(response => response.json())
    .then(data => {
        data.forEach(post => {
        nodes.push({ id: post.id, type: 'post', url: post.url });
        post.tags.forEach(tag => {
            if (!tags.has(tag)) {
            nodes.push({ id: tag, type: 'tag' });
            tags.add(tag);  // Add the tag to the set to avoid duplicates
            }
            links.push({ source: post.id, target: tag, label: tag });
        });
        });
        initializeGraph();  // Initialize the graph after the data is fetched
    })
    .catch(error => console.error("Error loading JSON:", error));
}

// Initialize the graph (D3 code)
function initializeGraph() {
  // Set up SVG and simulation
  const svg = d3.select("svg");
  const width = svg.node().getBoundingClientRect().width;
  const height = svg.node().getBoundingClientRect().height;

  const zoomBehavior = d3.zoom()
    .scaleExtent([0.3, 5]) // Set zoom-out and zoom-in limits
    .on("zoom", (event) => {
      container.attr("transform", event.transform); // Apply zoom transform
    });

  svg.call(zoomBehavior);

  const container = svg.append("g");

  // Create a force simulation to position the nodes
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2));  // Center the graph initially

  // Create link elements
  const link = container.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(links)
    .enter().append("line")
    .attr("class", "link");

  // Create node elements
  const node = container.append("g")
  .attr("class", "nodes")
  .selectAll("circle")
  .data(nodes)
  .enter().append("circle")
  .attr("class", d => d.type === 'post' ? "post-node" : "tag-node")
  .attr("r", d => d.type === 'post' ? 8 : 4)
  .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended))
  .on("click", (event, d) => {
    if (d.type === 'post' && d.url) {
      window.location.href = d.url;  // Redirect to the post URL
    }if (d.type === 'tag') {
      window.location.href = generateTagUrl(d.id);
    } 
  });

  // Add text labels for tag nodes
  const labels = container.append("g")
    .attr("class", "node-labels")
    .selectAll("text")
    .data(nodes)
    .enter().append("text")
    .attr("class", d => d.type === 'post' ? "post-label" : "tag-label")
    .attr("dy", "-10px") // Position the label above the node
    .text(d => d.id);

  // Update positions on each simulation tick
  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
    
    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
    
    labels
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });

  // Drag event handlers
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  // Update the fitGraphToContainer function to accept new dimensions
  function fitGraphToContainer(width, height) {
    const bounds = container.node().getBBox();
    const w = bounds.width;
    const h = bounds.height;
    const midX = bounds.x + w / 2;
    const midY = bounds.y + h / 2;

    if (w === 0 || h === 0) return; // Avoid division by zero

    // Calculate scale to fit the graph into view
    const scale = 0.9 / Math.max(w / width, h / height);
    const transform = d3.zoomIdentity
      .translate(width / 2 - scale * midX, height / 2 - scale * midY)
      .scale(scale);

    svg.transition().duration(500).call(zoomBehavior.transform, transform);
  }

  // Run the fit function once the simulation settles
  simulation.on("end", () => {
    fitGraphToContainer(width, height);
  });

  // Recenter the graph in smaller screens
  window.addEventListener('resize', () => {
    const newWidth = svg.node().getBoundingClientRect().width;
    const newHeight = svg.node().getBoundingClientRect().height;
    
    // Update the force simulation center to the new size
    simulation.force('center', d3.forceCenter(newWidth / 2, newHeight / 2));

    // Adjust zoom settings after resize
    svg.transition().duration(500).call(zoomBehavior.transform, zoomIdentity.translate(newWidth / 2, newHeight / 2).scale(1));
    
    // Fit the graph to the new size
    fitGraphToContainer(newWidth, newHeight);
  });

  // Ensure touch events for drag work on mobile devices
  svg.on("touchstart", function(event) {
    event.preventDefault(); // Prevent default to avoid zoom issues
  });
}

// Sluggify function
function slugify(str) {
  return str
    .toString()
    .toLowerCase()
    .normalize("NFD") // Remove accents
    .replace(/[\u0300-\u036f]/g, "") // Remove diacritics
    .replace(/[^a-z0-9 -]/g, '') // Remove invalid characters
    .replace(/\s+/g, '-') // Replace spaces with hyphens
    .replace(/-+/g, '-'); // Replace multiple hyphens with a single hyphen
}

// URL encode function
function urlEncode(str) {
  return encodeURIComponent(str);
}

// Process the tag (slugify, URL encode, and build the final URL)
function generateTagUrl(tag) {
  const sluggedTag = slugify(tag);
  const encodedTag = urlEncode(sluggedTag);
  const url = `/tags/${encodedTag}/`; // Prepend and append paths
  return url;
}

export { loadGraphData as graphInit };