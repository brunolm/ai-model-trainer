import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { spawn } from "node:child_process";

type ParsedArgs = {
  pythonPath: string;
  passthroughArgs: string[];
};

async function main() {
  const root = resolve(import.meta.dir, "../../../..");
  const runnerPath = resolve(root, "image-generation/examples/python/src/index.py");
  const defaultOutputPath = resolve(import.meta.dir, "../outputs/result.png");
  const { pythonPath, passthroughArgs } = parseArgs(Bun.argv.slice(2), root, defaultOutputPath);

  assertFile(runnerPath, "Python image example not found.");
  await runPython(pythonPath, runnerPath, passthroughArgs, process.cwd());
}

function parseArgs(args: string[], root: string, defaultOutputPath: string): ParsedArgs {
  const passthroughArgs: string[] = [];
  let pythonPath = defaultPythonPath(root);

  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];

    if (value === "--python") {
      const nextValue = args[index + 1];

      if (!nextValue) {
        throw new Error("--python requires a path to a Python executable");
      }

      pythonPath = nextValue;
      index += 1;
      continue;
    }

    passthroughArgs.push(value);
  }

  return {
    pythonPath,
    passthroughArgs: withDefaultOutput(passthroughArgs, defaultOutputPath),
  };
}

function defaultPythonPath(root: string) {
  const venvPython = resolve(root, ".venv/Scripts/python.exe");

  if (existsSync(venvPython)) {
    return venvPython;
  }

  return "python";
}

function withDefaultOutput(args: string[], defaultOutputPath: string) {
  if (hasOption(args, "--help") || hasOption(args, "-h") || hasOption(args, "--output")) {
    return args;
  }

  return [...args, "--output", defaultOutputPath];
}

function hasOption(args: string[], name: string) {
  return args.some((value) => value === name || value.startsWith(`${name}=`));
}

function assertFile(path: string, message: string) {
  if (existsSync(path)) {
    return;
  }

  throw new Error(`${message}\nMissing path: ${path}`);
}

async function runPython(pythonPath: string, runnerPath: string, args: string[], root: string) {
  const exitCode = await new Promise<number>((resolvePromise, reject) => {
    const child = spawn(pythonPath, [runnerPath, ...args], {
      cwd: root,
      env: process.env,
      stdio: "inherit",
      windowsHide: true,
    });

    child.on("error", reject);
    child.on("close", (code) => resolvePromise(code ?? 1));
  });

  if (exitCode === 0) {
    return;
  }

  throw new Error(`Python inference exited with code ${exitCode}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
