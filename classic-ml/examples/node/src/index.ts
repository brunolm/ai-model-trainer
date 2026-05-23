import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-node";

type VectorizerMetadata = {
  artifact_format: string;
  analyzer: "char" | "char_wb" | "word";
  lowercase: boolean;
  ngram_range: [number, number];
  norm: "l1" | "l2" | null;
  use_idf: boolean;
  sublinear_tf: boolean;
  token_pattern: string;
  vocabulary: Record<string, number>;
  idf: number[] | null;
  feature_count: number;
  classes: Array<number | string>;
  positive_label: number | string | null;
  onnx: {
    input_name: string;
    output_name: string;
  };
};

type PredictionResponse = {
  text: string;
  prediction: number | string;
  probabilities: Record<string, number>;
};

async function main() {
  const { text, threshold } = parseArgs(Bun.argv.slice(2));
  const paths = projectPaths();

  assertFile(paths.onnxPath, "ONNX model not found. Run `mise run train-profanity` from the repository root.");
  assertFile(paths.vectorizerPath, "Vectorizer metadata not found. Run `mise run train-profanity` from the repository root.");

  const vectorizer = loadVectorizer(paths.vectorizerPath);
  const prediction = await runPrediction(paths.onnxPath, vectorizer, text, threshold);
  printPrediction(prediction);
}

function parseArgs(args: string[]) {
  const textParts: string[] = [];
  let threshold: number | undefined;

  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];

    if (value === "--threshold") {
      threshold = Number(args[index + 1]);
      index += 1;
      continue;
    }

    textParts.push(value);
  }

  if (threshold !== undefined && Number.isNaN(threshold)) {
    throw new Error("--threshold must be a number");
  }

  return {
    text: textParts.join(" ").trim() || "hello friend",
    threshold,
  };
}

function projectPaths() {
  const root = resolve(import.meta.dir, "../../../..");

  return {
    root,
    onnxPath: resolve(root, "classic-ml/models/profanity-classifier.onnx"),
    vectorizerPath: resolve(root, "classic-ml/models/profanity-classifier-vectorizer.json"),
  };
}

function loadVectorizer(path: string) {
  const metadata = JSON.parse(readFileSync(path, "utf-8")) as VectorizerMetadata;

  if (metadata.artifact_format !== "ai-model-trainer-tfidf-vectorizer-v1") {
    throw new Error(`Unsupported vectorizer format: ${metadata.artifact_format}`);
  }

  return metadata;
}

async function runPrediction(
  onnxPath: string,
  vectorizer: VectorizerMetadata,
  text: string,
  threshold?: number,
) {
  const session = await ort.InferenceSession.create(onnxPath);
  const features = vectorizeText(text, vectorizer);
  const tensor = new ort.Tensor("float32", features, [1, vectorizer.feature_count]);
  const result = await session.run({ [vectorizer.onnx.input_name]: tensor });
  const logits = Array.from(result[vectorizer.onnx.output_name].data as Float32Array);
  const probabilities = softmax(logits);
  const prediction = predictedLabel(probabilities, vectorizer, threshold);

  return {
    text,
    prediction,
    probabilities: probabilityRows(probabilities, vectorizer.classes),
  };
}

function vectorizeText(text: string, vectorizer: VectorizerMetadata) {
  const features = new Float32Array(vectorizer.feature_count);
  const counts = new Map<number, number>();

  for (const token of analyze(text, vectorizer)) {
    const index = vectorizer.vocabulary[token];

    if (index === undefined) {
      continue;
    }

    counts.set(index, (counts.get(index) ?? 0) + 1);
  }

  for (const [index, count] of counts) {
    let value = vectorizer.sublinear_tf ? 1 + Math.log(count) : count;

    if (vectorizer.use_idf && vectorizer.idf) {
      value *= vectorizer.idf[index];
    }

    features[index] = value;
  }

  normalize(features, vectorizer.norm);
  return features;
}

function analyze(text: string, vectorizer: VectorizerMetadata) {
  const normalized = vectorizer.lowercase ? text.toLowerCase() : text;

  if (vectorizer.analyzer === "char_wb") {
    return charWbNgrams(normalized, vectorizer.ngram_range);
  }

  if (vectorizer.analyzer === "char") {
    return charNgrams(normalized, vectorizer.ngram_range);
  }

  return wordNgrams(normalized, vectorizer.ngram_range);
}

function charWbNgrams(text: string, [minN, maxN]: [number, number]) {
  const tokens: string[] = [];
  const words = text.trim().split(/\s+/).filter(Boolean);

  for (const word of words) {
    const padded = ` ${word} `;

    for (let size = minN; size <= maxN; size += 1) {
      if (padded.length < size) {
        tokens.push(padded);
        continue;
      }

      for (let index = 0; index <= padded.length - size; index += 1) {
        tokens.push(padded.slice(index, index + size));
      }
    }
  }

  return tokens;
}

function charNgrams(text: string, [minN, maxN]: [number, number]) {
  const tokens: string[] = [];

  for (let size = minN; size <= maxN; size += 1) {
    if (text.length < size) {
      continue;
    }

    for (let index = 0; index <= text.length - size; index += 1) {
      tokens.push(text.slice(index, index + size));
    }
  }

  return tokens;
}

function wordNgrams(text: string, [minN, maxN]: [number, number]) {
  const tokens = text.match(/[A-Za-z0-9_]{2,}/g) ?? [];
  const ngrams: string[] = [];

  for (let size = minN; size <= maxN; size += 1) {
    for (let index = 0; index <= tokens.length - size; index += 1) {
      ngrams.push(tokens.slice(index, index + size).join(" "));
    }
  }

  return ngrams;
}

function normalize(features: Float32Array, norm: VectorizerMetadata["norm"]) {
  if (!norm) {
    return;
  }

  const divisor = norm === "l1" ? l1Norm(features) : l2Norm(features);

  if (!divisor) {
    return;
  }

  for (let index = 0; index < features.length; index += 1) {
    features[index] /= divisor;
  }
}

function l1Norm(features: Float32Array) {
  let total = 0;

  for (const value of features) {
    total += Math.abs(value);
  }

  return total;
}

function l2Norm(features: Float32Array) {
  let total = 0;

  for (const value of features) {
    total += value * value;
  }

  return Math.sqrt(total);
}

function softmax(values: number[]) {
  const max = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function predictedLabel(probabilities: number[], vectorizer: VectorizerMetadata, threshold?: number) {
  if (threshold === undefined) {
    return vectorizer.classes[argMax(probabilities)];
  }

  if (vectorizer.classes.length !== 2) {
    throw new Error("--threshold is only supported for binary classifiers");
  }

  const positiveLabel = vectorizer.positive_label ?? vectorizer.classes[1];
  const positiveIndex = vectorizer.classes.findIndex((label) => String(label) === String(positiveLabel));
  const negativeIndex = positiveIndex === 0 ? 1 : 0;

  return probabilities[positiveIndex] >= threshold
    ? vectorizer.classes[positiveIndex]
    : vectorizer.classes[negativeIndex];
}

function probabilityRows(probabilities: number[], classes: Array<number | string>) {
  return Object.fromEntries(
    classes.map((label, index) => [`probability_${label}`, probabilities[index]]),
  );
}

function printPrediction(prediction: PredictionResponse) {
  console.log(`Text: ${prediction.text}`);
  console.log(`Prediction: ${labelName(prediction.prediction)}`);
  console.log("Probabilities:");

  for (const [label, value] of Object.entries(prediction.probabilities)) {
    console.log(`  ${label}: ${value.toFixed(4)}`);
  }

  console.log();
  console.log(JSON.stringify(prediction, null, 2));
}

function argMax(values: number[]) {
  let bestIndex = 0;

  for (let index = 1; index < values.length; index += 1) {
    if (values[index] <= values[bestIndex]) {
      continue;
    }

    bestIndex = index;
  }

  return bestIndex;
}

function labelName(label: number | string) {
  if (label === 1 || label === "1") {
    return "OFFENSIVE";
  }

  if (label === 0 || label === "0") {
    return "NOT_OFFENSIVE";
  }

  return String(label);
}

function assertFile(path: string, message: string) {
  if (existsSync(path)) {
    return;
  }

  throw new Error(`${message}\nMissing path: ${path}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
