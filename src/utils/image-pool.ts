/**
 * 图片链接池配置
 * 包含100张图片链接，用于博客文章的封面图
 */
export const IMAGE_POOL: string[] = [
	// 示例图片链接 - 请替换为实际的图片链接
	"https://i.pinimg.com/736x/a1/15/f4/a115f49dda91046eaf2759f93991c0c5.jpg",
];

/**
 * 默认图片链接（当网络图片加载失败时使用）
 */
export const DEFAULT_IMAGE = "/favicon/favicon-light-192.png";

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
