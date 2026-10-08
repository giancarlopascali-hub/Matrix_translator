import { MatrixPCA, PCA_BASIS_SCHEMA_VERSION } from '../src/lib/autoencoder.ts';
import { formatLatentCsv, parseFlattenedCsv, parseLatentCsv } from '../src/lib/matrixFormats.ts';

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

const training = binaryMatrices(6, 64);
const model = new MatrixPCA(8, 8, 16);
const fit = model.fit(training);
assert(fit.numComponents <= training.length - 1, 'PCA exported unsupported filler dimensions.');
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
});
const parsedLatents = parseLatentCsv(latentCsv);
assert(parsedLatents.length === training.length, 'Latent CSV row count changed after parsing.');
assert(parsedLatents[0].id === '100', 'Numeric matrix IDs were not preserved.');
assert(parsedLatents[0].detectedModelId === model.modelId, 'CSV model ID metadata was not preserved.');

for (let i = 0; i < training.length; i++) {
  const reconstructed = imported.decodeBinary(parsedLatents[i].z);
  assert(countDifferences(training[i], reconstructed) === 0, `Basis/CSV round-trip changed matrix ${i + 1}.`);
}

const training45 = binaryMatrices(5, 45 * 45, 0.2);
const model45 = new MatrixPCA(45, 45, 16);
const fit45 = model45.fit(training45);
assert(fit45.numComponents <= 4, '45×45 training data was padded beyond its supported rank.');
const imported45 = new MatrixPCA(1, 1, 1);
assert(imported45.loadBasis(model45.serializeBasis()), imported45.lastError || '45×45 basis did not reload.');
for (let i = 0; i < training45.length; i++) {
  const reconstructed = imported45.decodeBinary(model45.encode(training45[i]));
  assert(countDifferences(training45[i], reconstructed) === 0, `45×45 binary matrix ${i + 1} did not round-trip exactly.`);
}

const singleModel = new MatrixPCA(8, 8, 16);
const singleFit = singleModel.fit([training[0]]);
assert(singleFit.usedSyntheticAnchor, 'One-matrix fit did not use the hidden bootstrap anchor.');
assert(singleFit.numComponents === 1, 'One matrix plus one anchor should produce one active component.');
assert(singleModel.fittedCount === 1, 'Synthetic anchor was counted as an uploaded matrix.');
assert(countDifferences(training[0], singleModel.decodeBinary(singleModel.encode(training[0]))) === 0, 'Single binary matrix did not round-trip exactly.');
const singleBasis = JSON.parse(singleModel.serializeBasis());
assert(singleBasis.syntheticAnchorUsed === true, 'Basis did not record its single-matrix bootstrap mode.');
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
assert(lowRankFit.numComponents <= 2, 'Low-rank data was padded with arbitrary dimensions.');

const lossyModel = new MatrixPCA(8, 8, 2);
const lossyTraining = binaryMatrices(20, 64);
lossyModel.fit(lossyTraining);
assert(lossyModel.roundTripMetrics(lossyTraining).exactMatchRate < 1, 'Lossy PCA was incorrectly reported as exact.');

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
