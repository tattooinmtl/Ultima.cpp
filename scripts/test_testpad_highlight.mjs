// Regression test for the testpad editor's syntax highlighter.
//
// The highlighter lives inside ui/testpad/index.html because that file is
// bundled into the binary and may not reference anything external, so it has
// no unit-test harness of its own. This script extracts the Coder module from
// the HTML and checks that every language round-trips losslessly: stripping
// the spans and undoing the entities must give back the exact input. A pattern
// that drops text, duplicates it, or backtracks forever would otherwise only
// show up as a blank or scrambled editor.
//
// Run:  node scripts/test_testpad_highlight.mjs
//
// Not wired into CTest on purpose — the C++ build must not acquire a
// JavaScript toolchain dependency (Decision 02).

import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, '..', 'ui', 'testpad', 'index.html'), 'utf8');

const js = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));
const start = js.indexOf('const Coder = (() => {');
const end   = js.indexOf('// ---- Init ---');
if (start < 0 || end < 0) {
  console.error('could not locate the Coder module in ui/testpad/index.html');
  process.exit(1);
}

// The module only touches the DOM inside functions, so a stub $ is enough to
// import it and call the pure highlighting helpers.
const tmp = path.join(os.tmpdir(), 'ultima_coder_mod.mjs');
fs.writeFileSync(tmp, 'const $ = () => null;\n' + js.slice(start, end) + '\nexport default Coder;');
const { default: Coder } = await import(pathToFileURL(tmp).href);

const BT = String.fromCharCode(96);
const cases = [
  ['app.js', '// note\nconst x = ' + BT + 'tpl ${a}' + BT + ';\nlet s = \'hi\';\nfunction f(){ return 42; }'],
  ['a.py',   '# note\ndef f(x):\n    return "hi"\n'],
  ['a.css',  '/* c */\n.btn { color: #fff; margin: 4px; }'],
  ['a.html', '<!-- c --><div class="x" id=\'y\'>hi</div>'],
  ['a.json', '{"a": 1, "b": true, "c": null}'],
  ['a.md',   '# Title\nSome ' + BT + 'code' + BT + ' and **bold**.'],
  ['a.txt',  'plain text <b> & stuff'],
  // Pathological inputs. An unterminated string and a trailing backslash are
  // what hang a hand-rolled tokenizer; the big one guards against a pattern
  // that is accidentally quadratic.
  ['bad.js', 'const s = "unterminated\nconst t = 1; \\'],
  ['big.js', 'const x = 1;\n'.repeat(4000)],
];

let fail = 0;
for (const [name, src] of cases) {
  const lang = Coder.langFor(name);
  const t0 = Date.now();
  let out;
  try {
    out = Coder.highlight(src, lang);
  } catch (e) {
    console.log('THREW  ' + name + ': ' + e.message);
    fail++;
    continue;
  }
  const ms = Date.now() - t0;
  const plain = out.replace(/<[^>]+>/g, '')
                   .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const ok = plain === src;
  if (!ok) {
    fail++;
    console.log('MISMATCH ' + name);
    console.log('  got : ' + JSON.stringify(plain.slice(0, 100)));
    console.log('  want: ' + JSON.stringify(src.slice(0, 100)));
  }
  console.log((ok ? 'ok  ' : 'FAIL') + '  ' + name.padEnd(9) +
              ' lang=' + String(lang).padEnd(5) + ' ' + String(ms).padStart(4) + 'ms' +
              '  spans=' + (out.match(/<span/g) || []).length);
}

console.log(fail ? 'FAILURES: ' + fail : 'all highlighters lossless');
process.exit(fail ? 1 : 0);
