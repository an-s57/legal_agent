/** LexAgent Logo — 字母 L + A 几何组合，参考 Linear/OpenAI 风格 */
export default function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 28 28"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* L 的竖线 + A 的左斜线共用 */}
      <path
        d="M5 4 L5 24 L14 24 L14 20 L9 20 L9 4 Z"
        fill="#111827"
      />
      {/* A 的右斜线 + 横杠 */}
      <path
        d="M14 4 L23 24 L19 24 L16.5 18 L13.5 18 L14.5 20 L17 20 L14 13 L11 20 L13 20 L12 18 L9 18 L12 24 L8 24 Z M12 14 L16 14 L14 9 Z"
        fill="#111827"
        opacity="0"
      />
      {/* A 的右半部分，与 L 拼接 */}
      <path
        d="M14 4 L23 24 L19 24 L16.5 18 L13 18 L14.5 20.5 L17 20.5 L14 13.5 L11 20.5 L13.5 20.5 L15 18 L12 18 L9 24 L5 24 Z"
        fill="#111827"
        opacity="0"
      />
      {/* 简洁版：L 右侧加一个三角形代表 A */}
      <path
        d="M14 4 L23 24 L19 24 L16 17 L12 17 L9 24 L5 24 Z M14 10 L12.5 14 L15.5 14 Z"
        fill="#111827"
      />
    </svg>
  )
}
