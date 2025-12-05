---
title: "css modules原理学习"
date: 2023-03-16
permalink: /2023-03-16-css-modules/
---
## 整体思路

1. 识别 `:local` 包裹的类名，将其进行hash化，保证不污染全局
2. 将 `:local` 包裹的类名放在 `:export` 中，这个是icss的规范，算是css规范的超集。这样就相当于 `module.exports` ，外面使用时，可以通过 [`styles.xxx`](http://styles.xxx/) 的方式来拿到hash后的类名。

	```sass
	.guang {
	    color: blue;
	}
	._input_css_amSA5i__dong{
	    color: green;
	}
	._input_css_amSA5i__dongdong{
	    color: green;
	}
	._input_css_amSA5i__dongdongdong{
	    color: red;
	}
	@keyframes _input_css_amSA5i__guangguang {
	    from {
	        width: 0;
	    }
	    to {
	        width: 100px;
	    }
	}
	
	:export {
	  dong: _input_css_amSA5i__dong;
	  dongdong: _input_css_amSA5i__dongdong;
	  dongdongdong: _input_css_amSA5i__dongdongdong _input_css_amSA5i__dong _input_css_amSA5i__dongdong;
	  guangguang: _input_css_amSA5i__guangguang;
	}
	```


## AST 解析


前面提高的，识别 `:local()`和 `global()` 标记，并且对其进行hash化，就是通过AST实现的。


这些可以通过 astexplorer.net 来可视化的查看转义后的AST：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2023-03-16-css-modules/6202c5def9438de90826bc8d896e66f5.png)


整个转换的过程是：


具体代码这里 [https://github.com/QuarkGluonPlasma/postcss-plugin-exercize](https://github.com/QuarkGluonPlasma/postcss-plugin-exercize) ，或者展开下面：

<details>
<summary>代码</summary>

```typescript
const selectorParser = require("postcss-selector-parser");
function generateScopedName(name) {
    const randomStr = Math.random().toString(16).slice(2);
    return `_${randomStr}__${name}`;
};
const plugin = (options = {}) => {
    return {
        postcssPlugin: "my-postcss-modules-scope",
        Once(root, helpers) {
            const exports = {};
            function exportScopedName(name) {
                const scopedName = generateScopedName(name);

                exports[name] = exports[name] || [];

                if (exports[name].indexOf(scopedName) < 0) {
                    exports[name].push(scopedName);
                }
                return scopedName;
            }
            function localizeNode(node) {
                switch (node.type) {
                    case "selector":
                        node.nodes = node.map(localizeNode);
                        return node;
                    case "class":
                        return selectorParser.className({
                            value: exportScopedName(
                                node.value,
                                node.raws && node.raws.value ? node.raws.value : null
                            ),
                        });
                    case "id": {
                        return selectorParser.id({
                            value: exportScopedName(
                                node.value,
                                node.raws && node.raws.value ? node.raws.value : null
                            ),
                        });
                    }
                }
            }
            function traverseNode(node) {
                switch (node.type) {
                    case "root":
                    case "selector": {
                        node.each(traverseNode);
                        break;
                    }
                    case "id":
                    case "class":
                        exports[node.value] = [node.value];
                        break;
                    case "pseudo":
                        if (node.value === ":local") {
                            const selector = localizeNode(node.first, node.spaces);

                            node.replaceWith(selector);

                            return;
                        }
                }
                return node;
            }
            // 处理 :local 选择器
            root.walkRules((rule) => {
                const parsedSelector = selectorParser().astSync(rule);
                rule.selector = traverseNode(parsedSelector.clone()).toString();
                rule.walkDecls(/composes|compose-with/i, (decl) => {
                    const localNames = parsedSelector.nodes.map((node) => {
                        return node.nodes[0].first.first.value;
                    })
                    const classes = decl.value.split(/\s+/);
                    classes.forEach((className) => {
                        const global = /^global\(([^)]+)\)$/.exec(className);

                        if (global) {
                            localNames.forEach((exportedName) => {
                                exports[exportedName].push(global[1]);
                            });
                        } else if (Object.prototype.hasOwnProperty.call(exports, className)) {
                            localNames.forEach((exportedName) => {
                                exports[className].forEach((item) => {
                                    exports[exportedName].push(item);
                                });
                            });
                        } else {
                            throw decl.error(
                                `referenced class name "${className}" in ${decl.prop} not found`
                            );
                        }
                    });

                    decl.remove();
                });
            });
            // 处理 :local keyframes
            root.walkAtRules(/keyframes$/i, (atRule) => {
                const localMatch = /^:local\((.*)\)$/.exec(atRule.params);

                if (localMatch) {
                    atRule.params = exportScopedName(localMatch[1]);
                }
            });
            // 生成 :export rule
            const exportedNames = Object.keys(exports);

            if (exportedNames.length > 0) {
                const exportRule = helpers.rule({ selector: ":export" });

                exportedNames.forEach((exportedName) =>
                    exportRule.append({
                        prop: exportedName,
                        value: exports[exportedName].join(" "),
                        raws: { before: "\n  " },
                    })
                );
                root.append(exportRule);
            }
        },
    };
};
plugin.postcss = true;
module.exports = plugin;
```


</details>


## 参考文档

- [Untitled](https://www.notion.so/caa25f1f93c343cc95afbcd4d30bfd37)

[bookmark](https://github.com/css-modules/icss#specification)


[link_preview](https://github.com/camsong/blog/issues/5)


[bookmark](https://www.51cto.com/article/707429.html)


