/**
 * 图片链接池配置
 * 包含100张图片链接，用于博客文章的封面图
 */
export const IMAGE_POOL: string[] = [
	// 示例图片链接 - 请替换为实际的图片链接
	"https://picsum.photos/800/400?random=1",
	"https://picsum.photos/800/400?random=2",
	"https://picsum.photos/800/400?random=3",
	"https://picsum.photos/800/400?random=4",
	"https://picsum.photos/800/400?random=5",
	"https://picsum.photos/800/400?random=6",
	"https://picsum.photos/800/400?random=7",
	"https://picsum.photos/800/400?random=8",
	"https://picsum.photos/800/400?random=9",
	"https://picsum.photos/800/400?random=10",
	"https://picsum.photos/800/400?random=11",
	"https://picsum.photos/800/400?random=12",
	"https://picsum.photos/800/400?random=13",
	"https://picsum.photos/800/400?random=14",
	"https://picsum.photos/800/400?random=15",
	"https://picsum.photos/800/400?random=16",
	"https://picsum.photos/800/400?random=17",
	"https://picsum.photos/800/400?random=18",
	"https://picsum.photos/800/400?random=19",
	"https://picsum.photos/800/400?random=20",
	"https://picsum.photos/800/400?random=21",
	"https://picsum.photos/800/400?random=22",
	"https://picsum.photos/800/400?random=23",
	"https://picsum.photos/800/400?random=24",
	"https://picsum.photos/800/400?random=25",
	"https://picsum.photos/800/400?random=26",
	"https://picsum.photos/800/400?random=27",
	"https://picsum.photos/800/400?random=28",
	"https://picsum.photos/800/400?random=29",
	"https://picsum.photos/800/400?random=30",
	"https://picsum.photos/800/400?random=31",
	"https://picsum.photos/800/400?random=32",
	"https://picsum.photos/800/400?random=33",
	"https://picsum.photos/800/400?random=34",
	"https://picsum.photos/800/400?random=35",
	"https://picsum.photos/800/400?random=36",
	"https://picsum.photos/800/400?random=37",
	"https://picsum.photos/800/400?random=38",
	"https://picsum.photos/800/400?random=39",
	"https://picsum.photos/800/400?random=40",
	"https://picsum.photos/800/400?random=41",
	"https://picsum.photos/800/400?random=42",
	"https://picsum.photos/800/400?random=43",
	"https://picsum.photos/800/400?random=44",
	"https://picsum.photos/800/400?random=45",
	"https://picsum.photos/800/400?random=46",
	"https://picsum.photos/800/400?random=47",
	"https://picsum.photos/800/400?random=48",
	"https://picsum.photos/800/400?random=49",
	"https://picsum.photos/800/400?random=50",
	"https://picsum.photos/800/400?random=51",
	"https://picsum.photos/800/400?random=52",
	"https://picsum.photos/800/400?random=53",
	"https://picsum.photos/800/400?random=54",
	"https://picsum.photos/800/400?random=55",
	"https://picsum.photos/800/400?random=56",
	"https://picsum.photos/800/400?random=57",
	"https://picsum.photos/800/400?random=58",
	"https://picsum.photos/800/400?random=59",
	"https://picsum.photos/800/400?random=60",
	"https://picsum.photos/800/400?random=61",
	"https://picsum.photos/800/400?random=62",
	"https://picsum.photos/800/400?random=63",
	"https://picsum.photos/800/400?random=64",
	"https://picsum.photos/800/400?random=65",
	"https://picsum.photos/800/400?random=66",
	"https://picsum.photos/800/400?random=67",
	"https://picsum.photos/800/400?random=68",
	"https://picsum.photos/800/400?random=69",
	"https://picsum.photos/800/400?random=70",
	"https://picsum.photos/800/400?random=71",
	"https://picsum.photos/800/400?random=72",
	"https://picsum.photos/800/400?random=73",
	"https://picsum.photos/800/400?random=74",
	"https://picsum.photos/800/400?random=75",
	"https://picsum.photos/800/400?random=76",
	"https://picsum.photos/800/400?random=77",
	"https://picsum.photos/800/400?random=78",
	"https://picsum.photos/800/400?random=79",
	"https://picsum.photos/800/400?random=80",
	"https://picsum.photos/800/400?random=81",
	"https://picsum.photos/800/400?random=82",
	"https://picsum.photos/800/400?random=83",
	"https://picsum.photos/800/400?random=84",
	"https://picsum.photos/800/400?random=85",
	"https://picsum.photos/800/400?random=86",
	"https://picsum.photos/800/400?random=87",
	"https://picsum.photos/800/400?random=88",
	"https://picsum.photos/800/400?random=89",
	"https://picsum.photos/800/400?random=90",
	"https://picsum.photos/800/400?random=91",
	"https://picsum.photos/800/400?random=92",
	"https://picsum.photos/800/400?random=93",
	"https://picsum.photos/800/400?random=94",
	"https://picsum.photos/800/400?random=95",
	"https://picsum.photos/800/400?random=96",
	"https://picsum.photos/800/400?random=97",
	"https://picsum.photos/800/400?random=98",
	"https://picsum.photos/800/400?random=99",
	"https://picsum.photos/800/400?random=100",
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
