const fs = require('fs');
const path = require('path');
const targetPath = 'c:\\Users\\masca\\source\\repos\\_Mascari4615\\Mascari4615.github.io\\apps\\karmolab\\js\\toolbox.js';
const widgetsDir = 'c:\\Users\\masca\\source\\repos\\_Mascari4615\\Mascari4615.github.io\\apps\\karmolab\\js\\widgets';

if (!fs.existsSync(widgetsDir)) {
    fs.mkdirSync(widgetsDir, { recursive: true });
}

let content = fs.readFileSync(targetPath, 'utf8');

const widgetsParams = [
  { name: 'Hacker', id: 'hacker' },
  { name: 'Particle', id: 'particle' },
  { name: 'Bubble', id: 'bubble' },
  { name: 'Bounce', id: 'bounce' },
  { name: 'Morse', id: 'morse' },
  { name: 'Toast', id: 'toast' },
  { name: 'Darkroom', id: 'darkroom' },
  { name: 'Folder', id: 'folder' },
  { name: 'Countdown', id: 'countdown' },
  { name: 'Moon', id: 'moon' },
  { name: 'Password', id: 'password' },
  { name: 'ShyLink', id: 'shylink' },
  { name: 'News', id: 'news' },
  { name: 'Eyes', id: 'eyes' }
];

let appContentStart = content.indexOf('const PlaygroundApp = {');
let appContentEnd = content.indexOf('/* ===== 개별 도구 등록 ===== */');
if (appContentEnd === -1) appContentEnd = content.indexOf('/* ===== 개별 도구 등록');
let appContent = content.substring(appContentStart, appContentEnd);
let registerContent = content.substring(appContentEnd);

widgetsParams.forEach((w, i) => {
  const isLast = i === widgetsParams.length - 1;
  const nextTarget = isLast ? '    };\n})();' : 'build' + widgetsParams[i+1].name + '(container)';
  
  let fnStart = appContent.indexOf('build' + w.name + '(container)');
  if (fnStart === -1) {
    console.log(`Could not find build${w.name}`);
    return;
  }
  let fnEnd = appContent.indexOf(nextTarget, fnStart);
  if (fnEnd === -1 && isLast) {
    fnEnd = appContent.lastIndexOf('};\n})();');
    if (fnEnd === -1) fnEnd = appContent.lastIndexOf('    };');
  }
  
  let buildFn = appContent.substring(fnStart, fnEnd).trim();
  if (buildFn.endsWith(',')) buildFn = buildFn.substring(0, buildFn.length - 1);
  
  buildFn = buildFn.replace('build' + w.name + '(container)', 'function(container)');

  let searchId = `id: '${w.id}'`;
  let regStartInfo = registerContent.indexOf(searchId);
  if (regStartInfo === -1) {
    console.log(`Could not find register for ${w.id}`);
    return;
  }
  let regStart = registerContent.lastIndexOf('Toolbox.register(', regStartInfo);
  let regEnd = registerContent.indexOf('});', regStart) + 3;
  
  let regCall = registerContent.substring(regStart, regEnd);
  regCall = regCall.replace('PlaygroundApp.build' + w.name, buildFn);

  let outCode = '(function() {\n    ' + regCall.split('\n').join('\n    ') + '\n})();\n';
  fs.writeFileSync(path.join(widgetsDir, w.id + '.js'), outCode.replace(/    \n/g, '\n'));
  console.log('Wrote ' + w.id + '.js');
});

let newContent = content.substring(0, appContentStart);
newContent += content.substring(content.indexOf('/* ===== Bootstrap ===== */'));
fs.writeFileSync(targetPath, newContent);
console.log('toolbox.js updated.');
