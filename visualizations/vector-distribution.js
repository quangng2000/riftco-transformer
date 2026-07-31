import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DIMENSIONS = [3, 8, 16, 32, 64, 128, 768, 4096];
const COLORS = {
  query: new THREE.Color(0x2563eb),
  inside: new THREE.Color(0xf59e0b),
  outside: new THREE.Color(0x94a3b8),
  collision: new THREE.Color(0xdc2626),
  sphere: new THREE.Color(0x64748b),
};
const DEG = 180 / Math.PI;
const RAD = Math.PI / 180;

function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required element #${id}`);
  }
  return element;
}

const elements = {
  dimension: requiredElement("dimension"),
  dimensionOutput: requiredElement("dimension-output"),
  vectorCount: requiredElement("vector-count"),
  vectorCountOutput: requiredElement("vector-count-output"),
  angleBand: requiredElement("angle-band"),
  angleBandOutput: requiredElement("angle-band-output"),
  queryMode: requiredElement("query-mode"),
  pairsMode: requiredElement("pairs-mode"),
  resample: requiredElement("resample"),
  scene: requiredElement("vector-scene"),
  sceneBadge: requiredElement("scene-badge"),
  sceneNote: requiredElement("scene-note"),
  collisionLegend: requiredElement("collision-legend"),
  chart: requiredElement("angle-chart"),
  measurementBadge: requiredElement("measurement-badge"),
  expectedStat: requiredElement("expected-stat"),
  observedStat: requiredElement("observed-stat"),
  spreadStat: requiredElement("spread-stat"),
  takeaway: requiredElement("takeaway"),
};

const state = {
  dimension: 3,
  vectorCount: 96,
  halfWidth: 1,
  mode: "query",
  seed: 20260728,
  vectors: [],
  mappedVectors: [],
  queryAngles: [],
  pairResult: null,
  profile: null,
};

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianFactory(random) {
  let spare = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }

    let u = 0;
    let v = 0;
    while (u === 0) {
      u = random();
    }
    while (v === 0) {
      v = random();
    }

    const magnitude = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    spare = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  };
}

function sampleUnitVectors(count, dimension, seed) {
  const random = mulberry32(seed + dimension * 101 + count * 17);
  const gaussian = gaussianFactory(random);
  const vectors = [];

  for (let index = 0; index < count; index += 1) {
    const vector = new Float32Array(dimension);
    let squaredNorm = 0;

    for (let axis = 0; axis < dimension; axis += 1) {
      const value = gaussian();
      vector[axis] = value;
      squaredNorm += value * value;
    }

    const inverseNorm = 1 / Math.sqrt(squaredNorm);
    for (let axis = 0; axis < dimension; axis += 1) {
      vector[axis] *= inverseNorm;
    }
    vectors.push(vector);
  }

  return vectors;
}

function clampDot(value) {
  return Math.max(-1, Math.min(1, value));
}

function angleFromDot(dot) {
  return Math.acos(clampDot(dot)) * DEG;
}

function mapVectorToThreeDimensions(vector, dimension) {
  if (dimension === 3) {
    return new THREE.Vector3(vector[0], vector[1], vector[2]).normalize();
  }

  const queryDot = clampDot(vector[0]);
  const radius = Math.sqrt(Math.max(0, 1 - queryDot * queryDot));
  const planeX = vector[1] ?? 1;
  const planeY = vector[2] ?? 0;
  const planeNorm = Math.hypot(planeX, planeY) || 1;

  return new THREE.Vector3(
    queryDot,
    (radius * planeX) / planeNorm,
    (radius * planeY) / planeNorm,
  );
}

function computePairAngles(vectors, dimension) {
  const totalPairs = (vectors.length * (vectors.length - 1)) / 2;
  const angles = new Float32Array(totalPairs);
  let offset = 0;
  let worstAbsoluteDot = -1;
  let worstPair = [0, 1];
  let worstAngle = 90;

  for (let left = 0; left < vectors.length; left += 1) {
    for (let right = left + 1; right < vectors.length; right += 1) {
      let dot = 0;
      for (let axis = 0; axis < dimension; axis += 1) {
        dot += vectors[left][axis] * vectors[right][axis];
      }

      const angle = angleFromDot(dot);
      angles[offset] = angle;
      offset += 1;

      if (Math.abs(dot) > worstAbsoluteDot) {
        worstAbsoluteDot = Math.abs(dot);
        worstPair = [left, right];
        worstAngle = angle;
      }
    }
  }

  return {
    angles,
    worstPair,
    worstAngle,
    worstAbsoluteDot,
  };
}

function computeTheoreticalProfile(dimension, halfWidth) {
  const integrationStep = 0.025;
  let weightTotal = 0;
  let weightInside = 0;
  let weightedVariance = 0;

  for (let angle = integrationStep / 2; angle < 180; angle += integrationStep) {
    const sine = Math.sin(angle * RAD);
    const weight = Math.exp((dimension - 2) * Math.log(Math.max(sine, 1e-15)));
    weightTotal += weight;
    weightedVariance += weight * (angle - 90) ** 2;
    if (Math.abs(angle - 90) <= halfWidth) {
      weightInside += weight;
    }
  }

  const density = [];
  for (let angle = 0; angle <= 180; angle += 0.5) {
    const sine = Math.sin(angle * RAD);
    density.push({
      angle,
      value:
        angle === 0 || angle === 180
          ? 0
          : Math.exp((dimension - 2) * Math.log(sine)),
    });
  }

  return {
    probability: weightInside / weightTotal,
    spread: Math.sqrt(weightedVariance / weightTotal),
    density,
  };
}

function countInsideBand(angles, halfWidth) {
  let count = 0;
  for (const angle of angles) {
    if (Math.abs(angle - 90) <= halfWidth) {
      count += 1;
    }
  }
  return count;
}

function formatInteger(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatPercent(value) {
  if (value < 0.001) {
    return "<0.1%";
  }
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(3.15, 2.15, 3.15);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
elements.scene.querySelector(".loading")?.remove();
elements.scene.append(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;
controls.minDistance = 2.3;
controls.maxDistance = 7;

const sphere = new THREE.Mesh(
  new THREE.SphereGeometry(1, 30, 22),
  new THREE.MeshBasicMaterial({
    color: COLORS.sphere,
    transparent: true,
    opacity: 0.16,
    wireframe: true,
  }),
);
scene.add(sphere);

const origin = new THREE.Mesh(
  new THREE.SphereGeometry(0.035, 14, 10),
  new THREE.MeshBasicMaterial({ color: COLORS.query }),
);
scene.add(origin);

const queryArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(0, 0, 0),
  1.28,
  COLORS.query.getHex(),
  0.11,
  0.065,
);
scene.add(queryArrow);

const vectorGroup = new THREE.Group();
const bandGroup = new THREE.Group();
scene.add(bandGroup);
scene.add(vectorGroup);

function disposeObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material.dispose());
    } else {
      child.material?.dispose();
    }
  });
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    disposeObject(child);
  }
}

function createBandGeometry(halfWidth) {
  const rows = 4;
  const segments = 128;
  const xLimit = Math.sin(halfWidth * RAD);
  const positions = [];
  const indices = [];

  for (let row = 0; row <= rows; row += 1) {
    const x = -xLimit + (2 * xLimit * row) / rows;
    const radius = Math.sqrt(Math.max(0, 1 - x * x));
    for (let segment = 0; segment <= segments; segment += 1) {
      const phi = (segment / segments) * Math.PI * 2;
      positions.push(x, radius * Math.cos(phi), radius * Math.sin(phi));
    }
  }

  const rowWidth = segments + 1;
  for (let row = 0; row < rows; row += 1) {
    for (let segment = 0; segment < segments; segment += 1) {
      const a = row * rowWidth + segment;
      const b = a + rowWidth;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function createBoundaryCircle(x) {
  const radius = Math.sqrt(Math.max(0, 1 - x * x));
  const points = [];
  for (let segment = 0; segment < 128; segment += 1) {
    const phi = (segment / 128) * Math.PI * 2;
    points.push(
      new THREE.Vector3(x, radius * Math.cos(phi), radius * Math.sin(phi)),
    );
  }
  return new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({
      color: COLORS.inside,
      transparent: true,
      opacity: 0.75,
    }),
  );
}

function updateBand() {
  clearGroup(bandGroup);
  const xLimit = Math.sin(state.halfWidth * RAD);
  const band = new THREE.Mesh(
    createBandGeometry(state.halfWidth),
    new THREE.MeshBasicMaterial({
      color: COLORS.inside,
      transparent: true,
      opacity: 0.19,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  bandGroup.add(band);
  bandGroup.add(createBoundaryCircle(-xLimit));
  bandGroup.add(createBoundaryCircle(xLimit));
}

function colorForVector(index, angle) {
  if (
    state.mode === "pairs" &&
    state.dimension === 3 &&
    state.pairResult?.worstPair.includes(index)
  ) {
    return COLORS.collision;
  }
  return Math.abs(angle - 90) <= state.halfWidth
    ? COLORS.inside
    : COLORS.outside;
}

function updateSceneVectors() {
  clearGroup(vectorGroup);
  const rayPositions = [];
  const rayColors = [];
  const pointPositions = [];
  const pointColors = [];

  state.mappedVectors.forEach((vector, index) => {
    const color = colorForVector(index, state.queryAngles[index]);
    rayPositions.push(0, 0, 0, vector.x, vector.y, vector.z);
    rayColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    pointPositions.push(vector.x, vector.y, vector.z);
    pointColors.push(color.r, color.g, color.b);
  });

  const rayGeometry = new THREE.BufferGeometry();
  rayGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(rayPositions, 3),
  );
  rayGeometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(rayColors, 3),
  );
  const rays = new THREE.LineSegments(
    rayGeometry,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.38,
    }),
  );

  const pointGeometry = new THREE.BufferGeometry();
  pointGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(pointPositions, 3),
  );
  pointGeometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(pointColors, 3),
  );
  const points = new THREE.Points(
    pointGeometry,
    new THREE.PointsMaterial({
      size: 0.055,
      sizeAttenuation: true,
      vertexColors: true,
    }),
  );

  vectorGroup.add(rays);
  vectorGroup.add(points);
}

const resizeObserver = new ResizeObserver(([entry]) => {
  const { width, height } = entry.contentRect;
  if (width === 0 || height === 0) {
    return;
  }
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});
resizeObserver.observe(elements.scene);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function drawChart(angles) {
  const svg = elements.chart;
  svg.replaceChildren();

  const width = Math.max(300, Math.round(svg.clientWidth || 680));
  const compact = width < 460;
  const height = compact ? 330 : 360;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.height = `${height}px`;

  const title = svgElement("title");
  title.textContent = `Angle distribution in ${state.dimension} dimensions`;
  svg.append(title);

  const description = svgElement("desc");
  description.textContent =
    `The selected band is ${90 - state.halfWidth} to ` +
    `${90 + state.halfWidth} degrees. ${formatPercent(
      state.profile.probability,
    )} of random angles are expected inside it.`;
  svg.append(description);

  const margin = {
    top: 34,
    right: compact ? 12 : 18,
    bottom: 48,
    left: compact ? 46 : 54,
  };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (angle) => margin.left + (angle / 180) * plotWidth;
  const y = (value) => margin.top + (1 - value) * plotHeight;

  const bandLow = 90 - state.halfWidth;
  const bandHigh = 90 + state.halfWidth;
  svg.append(
    svgElement("rect", {
      x: x(bandLow),
      y: margin.top,
      width: Math.max(2, x(bandHigh) - x(bandLow)),
      height: plotHeight,
      class: "selected-band",
    }),
  );

  const tickStep = compact ? 45 : 30;
  for (let tick = 0; tick <= 180; tick += tickStep) {
    svg.append(
      svgElement("line", {
        x1: x(tick),
        x2: x(tick),
        y1: margin.top,
        y2: margin.top + plotHeight,
        class: "grid",
      }),
    );
    const label = svgElement("text", {
      x: x(tick),
      y: margin.top + plotHeight + 23,
      "text-anchor": "middle",
      class: "tick-label",
    });
    label.textContent = `${tick}°`;
    svg.append(label);
  }

  const binCount = 36;
  const bins = new Array(binCount).fill(0);
  for (const angle of angles) {
    const bin = Math.min(binCount - 1, Math.floor((angle / 180) * binCount));
    bins[bin] += 1;
  }
  const maximumBin = Math.max(1, ...bins);
  const binWidth = plotWidth / binCount;

  bins.forEach((count, index) => {
    const normalized = count / maximumBin;
    svg.append(
      svgElement("rect", {
        x: margin.left + index * binWidth + 1,
        y: y(normalized),
        width: Math.max(1, binWidth - 2),
        height: normalized * plotHeight,
        rx: 1,
        class: "histogram-bar",
      }),
    );
  });

  const curvePoints = state.profile.density.map((point) => [
    x(point.angle),
    y(point.value),
  ]);
  const linePath = curvePoints
    .map(
      ([pointX, pointY], index) =>
        `${index === 0 ? "M" : "L"} ${pointX.toFixed(2)} ${pointY.toFixed(2)}`,
    )
    .join(" ");
  const areaPath = `${linePath} L ${x(180)} ${y(0)} L ${x(0)} ${y(0)} Z`;
  svg.append(svgElement("path", { d: areaPath, class: "density-area" }));
  svg.append(svgElement("path", { d: linePath, class: "density-line" }));

  svg.append(
    svgElement("line", {
      x1: x(90),
      x2: x(90),
      y1: margin.top,
      y2: margin.top + plotHeight,
      class: "center-line",
    }),
  );
  svg.append(
    svgElement("line", {
      x1: x(bandLow),
      x2: x(bandHigh),
      y1: 22,
      y2: 22,
      class: "band-bracket",
    }),
  );
  const bandLabel = svgElement("text", {
    x: x(90),
    y: 16,
    "text-anchor": "middle",
    class: "band-label",
  });
  bandLabel.textContent = `${bandLow}°–${bandHigh}°`;
  svg.append(bandLabel);

  svg.append(
    svgElement("line", {
      x1: margin.left,
      x2: margin.left + plotWidth,
      y1: margin.top + plotHeight,
      y2: margin.top + plotHeight,
      class: "axis",
    }),
  );

  const xLabel = svgElement("text", {
    x: margin.left + plotWidth / 2,
    y: height - 8,
    "text-anchor": "middle",
    class: "axis-label",
  });
  xLabel.textContent = "angle θ";
  svg.append(xLabel);

  const yLabel = svgElement("text", {
    x: 16,
    y: margin.top + plotHeight / 2,
    transform: `rotate(-90 16 ${margin.top + plotHeight / 2})`,
    "text-anchor": "middle",
    class: "axis-label",
  });
  yLabel.textContent = "relative frequency";
  svg.append(yLabel);
}

function currentAngles() {
  if (state.mode === "query") {
    return state.queryAngles;
  }
  if (!state.pairResult) {
    state.pairResult = computePairAngles(state.vectors, state.dimension);
  }
  return state.pairResult.angles;
}

let chartResizeFrame = null;
const chartResizeObserver = new ResizeObserver(() => {
  if (!state.profile) {
    return;
  }
  if (chartResizeFrame !== null) {
    cancelAnimationFrame(chartResizeFrame);
  }
  chartResizeFrame = requestAnimationFrame(() => {
    chartResizeFrame = null;
    drawChart(currentAngles());
  });
});
chartResizeObserver.observe(elements.chart);

function updateText(angles) {
  const insideCount = countInsideBand(angles, state.halfWidth);
  const observedProbability = insideCount / angles.length;
  const bandLow = 90 - state.halfWidth;
  const bandHigh = 90 + state.halfWidth;

  elements.sceneBadge.textContent =
    state.dimension === 3 ? "Literal 3D" : `${state.dimension}D → 3D sketch`;
  elements.measurementBadge.textContent =
    state.mode === "query"
      ? `${formatInteger(angles.length)} q-to-feature angles`
      : `${formatInteger(angles.length)} feature pairs`;

  elements.expectedStat.textContent = formatPercent(state.profile.probability);
  elements.observedStat.textContent =
    `${formatInteger(insideCount)} / ${formatInteger(angles.length)} ` +
    `(${formatPercent(observedProbability)})`;
  elements.spreadStat.textContent = `σ ≈ ${state.profile.spread.toFixed(1)}°`;

  if (state.dimension === 3) {
    elements.sceneNote.textContent =
      "This sphere is literal: every arrow and every angle are shown exactly. " +
      "Drag to rotate it.";
  } else {
    elements.sceneNote.textContent =
      `Each ${state.dimension}D vector is placed so its angle to q is exact. ` +
      "Angles between feature arrows cannot all be preserved on a 3D screen.";
  }

  if (state.mode === "query") {
    elements.takeaway.textContent =
      `${formatPercent(state.profile.probability)} of random ` +
      `${state.dimension}D directions are expected between ${bandLow}° and ` +
      `${bandHigh}° from one query. Being near 90° to q does not mean the ` +
      "feature vectors are near 90° to each other.";
  } else {
    const allPairsPass = insideCount === angles.length;
    const worstDeviation = Math.abs(state.pairResult.worstAngle - 90);
    const literalThreeDimensionalLimit =
      state.dimension === 3
        ? " In 3D, at most three directions can all be this close to perpendicular."
        : "";
    elements.takeaway.textContent =
      `${formatInteger(insideCount)} of ${formatInteger(angles.length)} pairs ` +
      `land in the ${bandLow}°–${bandHigh}° band. All pairs pass: ` +
      `${allPairsPass ? "yes" : "no"}. The most overlapping pair is ` +
      `${worstDeviation.toFixed(1)}° away from a right angle.` +
      literalThreeDimensionalLimit;
  }

  elements.collisionLegend.hidden = !(
    state.mode === "pairs" && state.dimension === 3
  );
  elements.scene.setAttribute(
    "aria-label",
    `A sphere showing ${state.vectorCount} random vector directions. ` +
      `${countInsideBand(state.queryAngles, state.halfWidth)} are inside the ` +
      `${bandLow} to ${bandHigh} degree band relative to the blue query.`,
  );
  elements.chart.setAttribute(
    "aria-label",
    `Angle distribution for ${state.dimension} dimensions. ` +
      `${formatPercent(state.profile.probability)} are expected in the ` +
      `${bandLow} to ${bandHigh} degree band.`,
  );
}

function refreshDerivedViews() {
  state.profile = computeTheoreticalProfile(state.dimension, state.halfWidth);
  const angles = currentAngles();
  updateBand();
  updateSceneVectors();
  drawChart(angles);
  updateText(angles);
}

function regenerateVectors() {
  state.vectors = sampleUnitVectors(
    state.vectorCount,
    state.dimension,
    state.seed,
  );
  state.queryAngles = state.vectors.map((vector) => angleFromDot(vector[0]));
  state.mappedVectors = state.vectors.map((vector) =>
    mapVectorToThreeDimensions(vector, state.dimension),
  );
  state.pairResult = null;
  refreshDerivedViews();
}

function setMode(mode) {
  state.mode = mode;
  elements.queryMode.setAttribute("aria-pressed", String(mode === "query"));
  elements.pairsMode.setAttribute("aria-pressed", String(mode === "pairs"));
  refreshDerivedViews();
}

let regenerationFrame = null;
function scheduleRegeneration() {
  if (regenerationFrame !== null) {
    cancelAnimationFrame(regenerationFrame);
  }
  regenerationFrame = requestAnimationFrame(() => {
    regenerationFrame = null;
    regenerateVectors();
  });
}

elements.dimension.addEventListener("input", () => {
  state.dimension = DIMENSIONS[Number(elements.dimension.value)];
  elements.dimensionOutput.value = `${state.dimension}D`;
  scheduleRegeneration();
});

elements.vectorCount.addEventListener("input", () => {
  state.vectorCount = Number(elements.vectorCount.value);
  elements.vectorCountOutput.value = String(state.vectorCount);
  scheduleRegeneration();
});

elements.angleBand.addEventListener("input", () => {
  state.halfWidth = Number(elements.angleBand.value);
  elements.angleBandOutput.value = `${90 - state.halfWidth}°–${90 + state.halfWidth}°`;
  refreshDerivedViews();
});

elements.queryMode.addEventListener("click", () => setMode("query"));
elements.pairsMode.addEventListener("click", () => setMode("pairs"));
elements.resample.addEventListener("click", () => {
  state.seed += 1;
  regenerateVectors();
});

regenerateVectors();
