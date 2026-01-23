/**
 * 图片链接池配置
 * 从 blog 总目录下的 pinterest_image_urls.json 文件读取图片链接
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// 获取当前文件的目录路径
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 读取 JSON 文件（相对于项目根目录）
function loadImagePool(): string[] {
	try {
		// 从 src/utils 目录向上两级到达项目根目录
		const jsonPath = join(__dirname, "../../pinterest_image_urls.json");
		const fileContent = readFileSync(jsonPath, "utf-8");
		const imageUrls = JSON.parse(fileContent) as string[];
		
		// 验证是否为数组
		if (!Array.isArray(imageUrls)) {
			console.warn("pinterest_image_urls.json 应该包含一个字符串数组，使用默认值");
			return getDefaultImagePool();
		}
		
		// 过滤掉空值
		return imageUrls.filter((url) => url && typeof url === "string");
	} catch (error) {
		console.warn("无法读取 pinterest_image_urls.json，使用默认图片池:", error);
		return getDefaultImagePool();
	}
}

// 默认图片池（作为后备）
function getDefaultImagePool(): string[] {
	return [
		"https://i.pinimg.com/736x/a1/15/f4/a115f49dda91046eaf2759f93991c0c5.jpg",
	];
}

export const IMAGE_POOL: string[] = loadImagePool();

/**
 * 默认图片链接（当网络图片加载失败时使用）
 */
export const DEFAULT_IMAGE = "https://i.pinimg.com/736x/a1/15/f4/a115f49dda91046eaf2759f93991c0c5.jpg";

/**
 * 根据日期和文章ID生成稳定的随机图片链接
 *
 * @param postId 文章的唯一标识符（如slug或id）
 * @param date 日期字符串，格式为 YYYY-MM-DD，用于每天更新
 * @returns 图片链接
 */
export function getRandomImageForPost(postId: string, date?: string): string {
	const today = date || getTodayDateString();

	// 使用日期和文章ID生成一个稳定的种子
	const seed = hashString(`${today}-${postId}`);

	// 基于种子选择图片索引
	const imageIndex = seed % IMAGE_POOL.length;

	return IMAGE_POOL[imageIndex];
}

/**
 * 获取今天的日期字符串（YYYY-MM-DD格式）
 * 每天午夜12点后会自动更新
 */
function getTodayDateString(): string {
	const now = new Date();
	const year = now.getFullYear();
	const month = String(now.getMonth() + 1).padStart(2, "0");
	const day = String(now.getDate()).padStart(2, "0");
	return `${year}-${month}-${day}`;
}

/**
 * 简单的字符串哈希函数
 * 用于将字符串转换为数字种子
 */
function hashString(str: string): number {
	let hash = 0;
	for (let i = 0; i < str.length; i++) {
		const char = str.charCodeAt(i);
		hash = (hash << 5) - hash + char;
		hash = hash & hash; // Convert to 32bit integer
	}
	return Math.abs(hash);
}

/**
 * 检查图片链接是否为网络链接
 */
export function isNetworkImage(src: string): boolean {
	return src.startsWith("http://") || src.startsWith("https://");
}
