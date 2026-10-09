import { MatrixPCA, PCA_BASIS_SCHEMA_VERSION } from '../src/lib/autoencoder.ts';
import { validateBinaryDesign } from '../src/lib/designValidation.ts';
import { formatLatentCsv, formatValidatedLatentCsv, parseFlattenedCsv, parseLatentCsv } from '../src/lib/matrixFormats.ts';

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function assertThrows(fn: () => unknown, expectedText: string): void {
  try {
    fn();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    assert(message.includes(expectedText), `Expected error containing "${expectedText}", got "${message}".`);
    return;
  }
  throw new Error(`Expected an error containing "${expectedText}".`);
}

let seed = 0x12345678;
function random(): number {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 0x100000000;
}

function binaryMatrices(count: number, cells: number, density = 0.5): Float32Array[] {
  return Array.from(
    { length: count },
    () => Float32Array.from({ length: cells }, () => random() < density ? 1 : 0),
  );
}

function countDifferences(left: Float32Array, right: Float32Array): number {
  assert(left.length === right.length, 'Compared arrays must have the same length.');
  let differences = 0;
  for (let i = 0; i < left.length; i++) if (left[i] !== right[i]) differences++;
  return differences;
}

function binaryDesign(id: string, rows: number, cols: number, activeCells: Array<[number, number]>) {
  const data = new Float32Array(rows * cols);
  for (const [row, col] of activeCells) data[row * cols + col] = 1;
  return { id, rows, cols, data };
}

const training = binaryMatrices(6, 64);
const model = new MatrixPCA(8, 8, 16);
const fit = model.fit(training);
assert(fit.numComponents === 16, 'Encoder did not preserve the requested K=16 vector length.');
assert(fit.dataComponentCount <= training.length - 1, 'PCA reported more data-informed dimensions than the sample rank supports.');
assert(fit.completionComponentCount === 16 - fit.dataComponentCount, 'Fixed-length completion count is inconsistent.');
assert(model.hasDecodingBasis, 'Fitted model was not marked as a decoding basis.');

const metrics = model.roundTripMetrics(training);
assert(metrics.exactMatchRate === 1, 'Rank-supported training matrices should round-trip exactly.');

const basisJson = model.serializeBasis();
const imported = new MatrixPCA(1, 1, 1);
assert(imported.loadBasis(basisJson), imported.lastError || 'Serialized basis did not reload.');
assert(imported.modelId === model.modelId, 'Model ID changed after basis serialization.');

const latentItems = training.map((matrix, index) => ({
  id: String(index + 100),
  latent: model.encode(matrix),
}));
const latentCsv = formatLatentCsv(latentItems, true, {
  rows: 8,
  cols: 8,
  k: fit.numComponents,
  valueType: 'binary_01',
  modelId: model.modelId,
  schemaVersion: PCA_BASIS_SCHEMA_VERSION,
  dataComponentCount: fit.dataComponentCount,
});
const parsedLatents = parseLatentCsv(latentCsv);
assert(parsedLatents.length === training.length, 'Latent CSV row count changed after parsing.');
assert(parsedLatents[0].id === '100', 'Numeric matrix IDs were not preserved.');
assert(parsedLatents[0].detectedModelId === model.modelId, 'CSV model ID metadata was not preserved.');
assert(latentCsv.includes(`# data_informed_count: ${fit.dataComponentCount}`), 'CSV does not report the data-informed dimension count.');
assert(latentCsv.includes(`# completion_count: ${fit.completionComponentCount}`), 'CSV does not report the completion dimension count.');
assert(latentCsv.includes('z1=data_informed'), 'CSV does not flag its learned dimensions.');
if (fit.completionComponentCount > 0) {
  const firstCompletion = `z${fit.dataComponentCount + 1}`;
  assert(latentCsv.includes(`${firstCompletion}=completion`), 'CSV does not flag its completion dimensions.');
  assert(latentCsv.includes('# completion_reference_values:'), 'CSV does not provide completion reference values.');
}

for (let i = 0; i < training.length; i++) {
  const reconstructed = imported.decodeBinary(parsedLatents[i].z);
  assert(countDifferences(training[i], reconstructed) === 0, `Basis/CSV round-trip changed matrix ${i + 1}.`);
}

const training45 = binaryMatrices(5, 45 * 45, 0.2);
const model45 = new MatrixPCA(45, 45, 16);
const fit45 = model45.fit(training45);
assert(fit45.numComponents === 16, '45×45 encoder did not return the requested K=16 values.');
assert(fit45.dataComponentCount <= 4, '45×45 data-informed rank exceeds N-1.');
const imported45 = new MatrixPCA(1, 1, 1);
assert(imported45.loadBasis(model45.serializeBasis()), imported45.lastError || '45×45 basis did not reload.');
for (let i = 0; i < training45.length; i++) {
  const reconstructed = imported45.decodeBinary(model45.encode(training45[i]));
  assert(countDifferences(training45[i], reconstructed) === 0, `45×45 binary matrix ${i + 1} did not round-trip exactly.`);
}

const singleModel = new MatrixPCA(8, 8, 16);
const singleFit = singleModel.fit([training[0]]);
assert(singleFit.usedSyntheticAnchor, 'One-matrix fit did not use the hidden bootstrap anchor.');
assert(singleFit.numComponents === 16, 'Single-matrix encoding did not preserve requested K=16.');
assert(singleFit.dataComponentCount === 1 && singleFit.completionComponentCount === 15, 'Single-matrix fixed-length basis composition is incorrect.');
assert(singleModel.encode(training[0]).length === 16, 'Single-matrix latent vector does not contain K=16 values.');
assert(singleModel.fittedCount === 1, 'Synthetic anchor was counted as an uploaded matrix.');
assert(countDifferences(training[0], singleModel.decodeBinary(singleModel.encode(training[0]))) === 0, 'Single binary matrix did not round-trip exactly.');
const singleBasis = JSON.parse(singleModel.serializeBasis());
assert(singleBasis.syntheticAnchorUsed === true, 'Basis did not record its single-matrix bootstrap mode.');
const singleCsv = formatLatentCsv([{
  id: 'single',
  latent: singleModel.encode(training[0]),
}], true, {
  rows: 8,
  cols: 8,
  k: 16,
  valueType: 'binary_01',
  modelId: singleModel.modelId,
  schemaVersion: PCA_BASIS_SCHEMA_VERSION,
  dataComponentCount: singleFit.dataComponentCount,
  syntheticAnchorUsed: true,
});
const singleCsvDataRow = singleCsv.trim().split(/\r?\n/).at(-1) || '';
assert(!/[eE][+-]?\d+/.test(singleCsvDataRow), 'Latent CSV exposed floating-point noise in scientific notation.');
assert(singleCsv.includes('# bootstrap_direction: z1'), 'Single-matrix CSV did not identify its bootstrap direction.');
assert(singleCsv.includes('z2=completion'), 'Single-matrix CSV did not flag completion dimensions.');
const parsedSingleCsv = parseLatentCsv(singleCsv);
assert(parsedSingleCsv.length === 1 && parsedSingleCsv[0].z.length === 16, 'Clean single-matrix CSV did not preserve K=16.');
assert(countDifferences(training[0], singleModel.decodeBinary(parsedSingleCsv[0].z)) === 0, 'Readable CSV formatting changed the single-matrix round trip.');
const importedSingle = new MatrixPCA(1, 1, 1);
assert(importedSingle.loadBasis(JSON.stringify(singleBasis)), importedSingle.lastError || 'Single-matrix basis did not reload.');
assert(importedSingle.usedSyntheticAnchor, 'Imported basis lost the synthetic-anchor metadata.');
assert(countDifferences(training[0], importedSingle.decodeBinary(singleModel.encode(training[0]))) === 0, 'Imported single-matrix basis changed the matrix.');

const zeroMatrix = new Float32Array(64);
const zeroModel = new MatrixPCA(8, 8, 4);
zeroModel.fit([zeroMatrix]);
assert(countDifferences(zeroMatrix, zeroModel.decodeBinary(zeroModel.encode(zeroMatrix))) === 0, 'All-zero single matrix did not round-trip exactly.');

const singleContinuous = Float32Array.from([0.1, -0.2, 0.3, 0.4]);
const singleContinuousModel = new MatrixPCA(2, 2, 4);
singleContinuousModel.fit([singleContinuous]);
const singleContinuousDecoded = singleContinuousModel.decode(singleContinuousModel.encode(singleContinuous));
for (let i = 0; i < singleContinuous.length; i++) {
  assert(Math.abs(singleContinuous[i] - singleContinuousDecoded[i]) < 1e-6, 'Single continuous matrix did not round-trip accurately.');
}

const twoRealFit = singleModel.fit(training.slice(0, 2));
assert(!twoRealFit.usedSyntheticAnchor && !singleModel.usedSyntheticAnchor, 'Synthetic anchor was not discarded after a second real matrix was loaded.');

const failedRefit = new MatrixPCA(8, 8, 4);
failedRefit.fit(training.slice(0, 4));
assertThrows(() => failedRefit.fit([]), 'At least one matrix');
assert(!failedRefit.hasDecodingBasis, 'A failed refit left a stale basis active.');
assertThrows(() => imported.decode([0.5]), 'requires exactly K');
assertThrows(() => imported.decode(new Array(imported.k).fill(1.1)), 'inside [0,1]');

const lowRankModel = new MatrixPCA(8, 8, 16);
const lowRankFit = lowRankModel.fit(training.slice(0, 3));
assert(lowRankFit.numComponents === 16, 'Low-rank data did not retain the requested fixed K.');
assert(lowRankFit.dataComponentCount <= 2, 'Low-rank data reported too many learned components.');

const fixedEightModel = new MatrixPCA(45, 45, 8);
const fixedEightFit = fixedEightModel.fit(training45.slice(0, 2));
assert(fixedEightFit.numComponents === 8, 'Requested K=8 did not produce eight basis components.');
assert(fixedEightModel.encode(training45[0]).length === 8, 'Requested K=8 did not produce eight latent values.');
const fixedEightBasis = JSON.parse(fixedEightModel.serializeBasis());
assert(fixedEightBasis.k === 8 && fixedEightBasis.components.length === 8, 'K=8 basis JSON has the wrong component count.');
assert(fixedEightBasis.dataComponentCount === fixedEightFit.dataComponentCount, 'Basis JSON lost the data/completion dimension boundary.');
for (let k = 0; k < fixedEightBasis.components.length; k++) {
  const component = fixedEightBasis.components[k] as number[];
  const norm = Math.sqrt(component.reduce((sum, value) => sum + value * value, 0));
  assert(Math.abs(norm - 1) < 1e-8, `K=8 basis component ${k + 1} is not normalized.`);
  for (let previous = 0; previous < k; previous++) {
    const dot = component.reduce(
      (sum, value, index) => sum + value * fixedEightBasis.components[previous][index],
      0,
    );
    assert(Math.abs(dot) < 1e-8, `K=8 basis components ${previous + 1} and ${k + 1} are not orthogonal.`);
  }
}
const fixedEightImported = new MatrixPCA(1, 1, 1);
assert(fixedEightImported.loadBasis(JSON.stringify(fixedEightBasis)), fixedEightImported.lastError || 'K=8 basis did not reload.');
assert(countDifferences(training45[0], fixedEightImported.decodeBinary(fixedEightModel.encode(training45[0]))) === 0, 'K=8 basis/latent pair changed its source matrix.');

const lossyModel = new MatrixPCA(8, 8, 2);
const lossyTraining = binaryMatrices(20, 64);
lossyModel.fit(lossyTraining);
assert(lossyModel.roundTripMetrics(lossyTraining).exactMatchRate < 1, 'Lossy PCA was incorrectly reported as exact.');

const topCornerBridge: Array<[number, number]> = [
  [0, 0], [0, 1], [1, 0], [1, 1],
  [0, 4], [0, 5], [1, 4], [1, 5],
  [1, 2], [1, 3],
];
const passingDesign = validateBinaryDesign(binaryDesign('passing', 6, 6, topCornerBridge));
assert(passingDesign.validation === 1, 'A connected design with two corner blocks did not pass.');
assert(passingDesign.componentCount === 1, 'The connected design reported more than one active component.');
assert(passingDesign.significantComponentCount === 1, 'The main connected region was not retained.');
assert(passingDesign.qualifyingCorners.join(',') === 'top_left,top_right', 'The qualifying corners were identified incorrectly.');

const disconnectedCorners = validateBinaryDesign(binaryDesign('disconnected', 6, 6, topCornerBridge.slice(0, 8)));
assert(disconnectedCorners.validation === 1, 'Disconnected 2x2 corner islands were not ignored.');
assert(disconnectedCorners.componentCount === 2, 'Disconnected corner islands reported the wrong component count.');
assert(disconnectedCorners.ignoredSmallComponentCount === 2, 'Disconnected 2x2 corner islands were not classified as small.');
assert(disconnectedCorners.significantComponentCount === 0, 'Small corner islands were incorrectly retained as large regions.');

const mainRegionWithSmallIsland = validateBinaryDesign(binaryDesign('small-island', 6, 6, [
  ...topCornerBridge,
  [5, 2], [5, 3],
]));
assert(mainRegionWithSmallIsland.validation === 1, 'A small disconnected island incorrectly invalidated the design.');
assert(mainRegionWithSmallIsland.significantComponentCount === 1, 'The validator did not retain exactly one larger region.');
assert(mainRegionWithSmallIsland.ignoredSmallComponentCount === 1, 'The small disconnected island was not ignored.');

const wideTopCornerBridge: Array<[number, number]> = [
  [0, 0], [0, 1], [1, 0], [1, 1],
  [0, 6], [0, 7], [1, 6], [1, 7],
  [1, 2], [1, 3], [1, 4], [1, 5],
];
const threeCellLine = validateBinaryDesign(binaryDesign('three-cell-line', 8, 8, [
  ...wideTopCornerBridge,
  [4, 3], [5, 3], [6, 3],
]));
assert(threeCellLine.validation === 1, 'A disconnected three-cell line incorrectly invalidated the design.');
assert(threeCellLine.ignoredSmallComponentCount === 1, 'A disconnected three-cell line was not ignored.');

const fourCellLine = validateBinaryDesign(binaryDesign('four-cell-line', 8, 8, [
  ...wideTopCornerBridge,
  [3, 3], [4, 3], [5, 3], [6, 3],
]));
assert(fourCellLine.validation === 1, 'A disconnected four-cell line incorrectly invalidated the design.');
assert(fourCellLine.ignoredSmallComponentCount === 1, 'A disconnected four-cell line was not ignored.');

const fourCellStaggered = validateBinaryDesign(binaryDesign('four-cell-staggered', 8, 8, [
  ...wideTopCornerBridge,
  [4, 2], [4, 3], [5, 3], [5, 4],
]));
assert(fourCellStaggered.validation === 1, 'A disconnected staggered four-cell pattern incorrectly invalidated the design.');
assert(fourCellStaggered.ignoredSmallComponentCount === 1, 'A disconnected staggered four-cell pattern was not ignored.');

const multipleLargeRegions = validateBinaryDesign(binaryDesign('large-island', 8, 8, [
  ...wideTopCornerBridge,
  [3, 3], [4, 3], [5, 3], [6, 3], [7, 3],
]));
assert(multipleLargeRegions.validation === 0, 'A disconnected five-cell region did not invalidate the design.');
assert(multipleLargeRegions.significantComponentCount === 2, 'The validator did not retain both larger disconnected regions.');
assert(multipleLargeRegions.ignoredSmallComponentCount === 0, 'A five-cell disconnected region was incorrectly ignored.');

const diagonalCornerBridge = validateBinaryDesign(binaryDesign('diagonal-corners', 6, 6, [
  [0, 0], [0, 1], [1, 0], [1, 1],
  [4, 4], [4, 5], [5, 4], [5, 5],
  [1, 2], [1, 3], [1, 4], [2, 4], [3, 4],
]));
assert(diagonalCornerBridge.validation === 1, 'Connected blocks in diagonally opposite corners did not pass.');
assert(diagonalCornerBridge.qualifyingCorners.join(',') === 'top_left,bottom_right', 'Opposite corner blocks were identified incorrectly.');

const oneCorner = validateBinaryDesign(binaryDesign('one-corner', 6, 6, [
  [0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3],
]));
assert(oneCorner.connectivityPassed && oneCorner.validation === 0, 'A connected design with only one corner block incorrectly passed.');

const diagonalOnly = validateBinaryDesign(binaryDesign('diagonal', 6, 6, [
  [0, 0], [0, 1], [1, 0], [1, 1],
  [2, 2], [3, 3],
  [4, 4], [4, 5], [5, 4], [5, 5],
]));
assert(diagonalOnly.validation === 1, 'Disconnected regions fitting within 2x2 were not ignored.');
assert(diagonalOnly.componentCount === diagonalOnly.ignoredSmallComponentCount, 'A small diagonal island was incorrectly retained.');

const singlePhysicalCornerBlock = validateBinaryDesign(binaryDesign('tiny', 2, 2, [
  [0, 0], [0, 1], [1, 0], [1, 1],
]));
assert(singlePhysicalCornerBlock.validation === 0, 'One physical 2x2 block was counted as multiple distinct corners.');

const nonBinaryDesign = validateBinaryDesign({
  id: 'non-binary',
  rows: 2,
  cols: 2,
  data: Float32Array.from([1, 1, 1, 0.5]),
});
assert(!nonBinaryDesign.binaryValuesOnly && nonBinaryDesign.validation === 0, 'A non-binary matrix passed binary design validation.');

const flattened = [
  'matrix_id,f0,f1,f2,f3',
  '7,0,1,1,0',
  '8,1,0,0,1',
].join('\n');
const flattenedMatrices = parseFlattenedCsv(flattened, 2, 2);
assert(flattenedMatrices.length === 2, 'Flattened CSV header became a bogus matrix.');
assert(flattenedMatrices[0].id === '7', 'Numeric flattened-matrix ID was not preserved.');
const headerlessFlattened = parseFlattenedCsv('99,0,1,1,0', 2, 2);
assert(headerlessFlattened.length === 1 && headerlessFlattened[0].id === '99', 'Headerless numeric matrix ID was not preserved.');

const optimizerCsv = [
  '# dimensions: 8x8',
  'matrix_id,z1,z2,objective',
  '123,0.25,0.75,999',
].join('\n');
const optimizerRows = parseLatentCsv(optimizerCsv);
assert(optimizerRows.length === 1 && optimizerRows[0].z.length === 2, 'Non-latent BO columns were parsed as z values.');
assert(optimizerRows[0].id === '123', 'Numeric BO row ID was parsed as a latent coordinate.');

const validatedOptimizerCsv = formatValidatedLatentCsv(optimizerCsv, optimizerRows, [1], 0.5);
assert(validatedOptimizerCsv.includes('matrix_id,z1,z2,objective,validation'), 'Validation export did not append its column to the source header.');
assert(validatedOptimizerCsv.includes('123,0.25,0.75,999,1'), 'Validation export changed or discarded optimizer row values.');
assert(validatedOptimizerCsv.includes('# validation_rule: design_validation_v3'), 'Validation export did not use the current rule version.');
assert(validatedOptimizerCsv.includes('# validation_connectivity: orthogonal_4'), 'Validation export omitted its connectivity contract.');
assert(validatedOptimizerCsv.includes('# validation_ignored_islands: bounding_box_at_most_2x2_or_at_most_4_cells'), 'Validation export omitted its ignored-island contract.');
assert(validatedOptimizerCsv.includes('# validation_corner_requirement: at_least_2_distinct_edge_anchored_2x2'), 'Validation export omitted its corner contract.');
const reloadedValidatedRows = parseLatentCsv(validatedOptimizerCsv);
assert(reloadedValidatedRows.length === 1, 'Validated latent CSV could not be loaded by the decoder.');
assert(reloadedValidatedRows[0].z[0] === 0.25 && reloadedValidatedRows[0].z[1] === 0.75, 'Validated latent CSV changed latent coordinates.');

const revalidatedOptimizerCsv = formatValidatedLatentCsv(validatedOptimizerCsv, reloadedValidatedRows, [0], 0.5);
const revalidatedHeader = revalidatedOptimizerCsv.split(/\r?\n/).find(line => line.startsWith('matrix_id,')) || '';
assert(revalidatedHeader.split(',').filter(column => column === 'validation').length === 1, 'Repeated validation export duplicated the validation column.');
assert(revalidatedOptimizerCsv.includes('123,0.25,0.75,999,0'), 'Repeated validation export did not replace the old result.');

const headerlessLatentCsv = 'candidate,0.2,0.8';
const headerlessRows = parseLatentCsv(headerlessLatentCsv);
const validatedHeaderlessCsv = formatValidatedLatentCsv(headerlessLatentCsv, headerlessRows, [0], 0.5);
assert(validatedHeaderlessCsv.includes('matrix_id,z1,z2,validation'), 'Headerless validation export did not create a canonical header.');
assert(validatedHeaderlessCsv.includes('candidate,0.2,0.8,0'), 'Headerless validation export changed its row identity or values.');

const declaredKRows = parseLatentCsv([
  '# k: 3',
  'matrix_id,z1,z2',
  'candidate,0.25,0.75',
].join('\n'));
assert(declaredKRows[0].detectedK === 3, 'Declared CSV K metadata was silently overwritten.');

const tampered = JSON.parse(basisJson);
tampered.zMin[0] += 0.001;
const tamperedModel = new MatrixPCA(1, 1, 1);
assert(!tamperedModel.loadBasis(JSON.stringify(tampered)), 'Tampered basis passed its model ID check.');

const legacy = JSON.parse(basisJson);
delete legacy.schemaVersion;
delete legacy.algorithm;
delete legacy.modelId;
delete legacy.valueType;
const legacyModel = new MatrixPCA(1, 1, 1);
assert(!legacyModel.loadBasis(JSON.stringify(legacy)), 'An incompatible legacy basis was silently accepted.');
assert(legacyModel.lastError?.includes('old or unsupported'), 'Legacy basis rejection did not explain how to migrate.');

const continuousModel = new MatrixPCA(2, 2, 2);
continuousModel.fit([
  Float32Array.from([0.1, 0.2, 0.3, 0.4]),
  Float32Array.from([0.4, 0.3, 0.2, 0.1]),
  Float32Array.from([0.2, 0.4, 0.1, 0.3]),
]);
assert(JSON.parse(continuousModel.serializeBasis()).valueType === 'continuous_numeric', 'Continuous value type was mislabeled in the basis.');

const pythonAdapter = model45.generatePythonScript();
assert(!pythonAdapter.includes('def fit('), 'Python adapter can still refit a divergent model.');
assert(pythonAdapter.includes('SUPPORTED_SCHEMA_VERSION = 2'), 'Python adapter does not enforce the basis contract.');
assert(pythonAdapter.includes('Matrix dimension: 45 x 45'), 'Python adapter contains stale dimensions.');

const storage = new Map<string, string>();
(globalThis as any).window = {
  localStorage: {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
    removeItem: (key: string) => storage.delete(key),
  },
};
storage.set('pca_basis_latest', basisJson);
const differentShape = new MatrixPCA(9, 9, 5);
assert(differentShape.rows === 9 && differentShape.cols === 9, 'A stale latest basis changed the requested dimensions.');
assert(!differentShape.hasDecodingBasis, 'A stale latest basis was silently activated.');
delete (globalThis as any).window;

console.log('All matrix PCA round-trip tests passed.');
