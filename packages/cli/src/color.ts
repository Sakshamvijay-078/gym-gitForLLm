const isTTY = process.stdout.isTTY;

function wrap(code: string) {
  return (s: string) => (isTTY ? `\x1b[${code}m${s}\x1b[0m` : s);
}

export const color = {
  green: wrap("32"),
  red: wrap("31"),
  yellow: wrap("33"),
  dim: wrap("2"),
  bold: wrap("1"),
};
