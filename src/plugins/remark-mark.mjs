import { visit } from "unist-util-visit";

export default function remarkMark() {
	return (tree, file) => {
		visit(tree, "text", (node, index, parent) => {
			if (!node.value.includes("==")) return;

			const parts = node.value.split(/(==)/g);
			const newNodes = [];
			let inMark = false;

			for (const part of parts) {
				if (part === "==") {
					if (inMark) {
						newNodes.push({ type: "html", value: "</mark>" });
					} else {
						newNodes.push({ type: "html", value: "<mark>" });
					}
					inMark = !inMark;
				} else if (part.length > 0) {
					newNodes.push({ type: "text", value: part });
				}
			}

			// 替换当前节点
			parent.children.splice(index, 1, ...newNodes);

			// 返回索引，让 visit 继续处理新插入的节点
			return index + newNodes.length;
		});
	};
}
