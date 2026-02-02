/**
 * GitHub Light 风格自定义主题（极简高级）
 * 背景 #FFFFFF，主字体 #24292E
 * 注释 #6A737D，关键字 #D73A49，字符串 #032F62，数字 #005CC5，函数名 #6F42C1
 */
export const githubLightCustomTheme = {
	name: "github-light-custom",
	type: "light",
	bg: "#F6F8FA",
	fg: "#24292E",
	settings: [
		// 兜底默认：主字体
		{ scope: ["source", "text", "meta", "support.other", "entity.other"], settings: { foreground: "#24292E" } },
		// 注释 #6A737D
		{ scope: ["comment", "comment.line", "comment.block", "comment.block.html", "punctuation.definition.comment"], settings: { foreground: "#6A737D" } },
		// 关键字 #D73A49
		{ scope: ["keyword", "keyword.control", "keyword.operator", "keyword.other", "keyword.operator.new"], settings: { foreground: "#D73A49" } },
		// 字符串 #032F62
		{ scope: ["string", "string.quoted", "string.quoted.double", "string.quoted.single"], settings: { foreground: "#032F62" } },
		// 数字 #005CC5
		{ scope: ["constant.numeric", "constant.language", "constant.character", "constant.other"], settings: { foreground: "#005CC5" } },
		// 函数名 #6F42C1
		{ scope: ["entity.name.function", "support.function", "meta.function-call", "variable.function"], settings: { foreground: "#6F42C1" } },
		// 标签/类型/类名：深绿
		{ scope: ["entity.name.tag", "entity.name.tag.html", "entity.name.type", "support.type", "support.class", "entity.name.type.class"], settings: { foreground: "#116329" } },
		// 变量/参数：主色
		{ scope: ["variable", "variable.parameter", "variable.other", "variable.readwrite", "meta.parameter"], settings: { foreground: "#24292E" } },
		// 标点/括号：主色
		{ scope: ["punctuation", "meta.brace", "punctuation.definition.tag", "punctuation.separator"], settings: { foreground: "#24292E" } },
		// HTML/JSX 属性名、属性值、class 等：主色，避免浅灰
		{ scope: ["meta.tag", "meta.tag.inline", "meta.attribute", "meta.attribute.with-value", "entity.other.attribute-name", "entity.other.attribute-name.html", "meta.attribute.with-value.html", "support.constant", "constant.other.symbol", "meta.tag.metadata", "entity.name.tag.custom"], settings: { foreground: "#24292E" } },
		{ scope: ["markup", "markup.underline", "markup.bold", "markup.italic", "markup.heading", "markup.list", "meta.embedded"], settings: { foreground: "#24292E" } },
		{ scope: ["storage.type", "storage.modifier", "entity.name.section"], settings: { foreground: "#24292E" } },
	],
};
