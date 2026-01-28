import { visit } from "unist-util-visit";

export default function remarkMark() {
	return (tree) => {
		visit(tree, "text", (node, index, parent) => {
			if (!node.value || !parent) return;

			// Replace ==text== with <mark>text</mark>
			if (node.value.includes("==")) {
				node.type = "html";
				node.value = node.value.replace(/==(.+?)==/g, "<mark>$1</mark>");
			}
		});
	};
}
