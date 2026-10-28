"""trainplat_jd193 —— 用规则在大模型训练平台工程师·分块策略对召回的影响里定位可疑位置。

用一组明示的正则规则扫描输入，报告大模型训练平台工程师·分块策略对召回的影响相关的命中行与片段

定位：
  规则表公开可读，每条命中都能对应到一条明示的规则

岗位对应：
  目标岗位　华为 · 专家大模型训练平台工程师
  技能主线　PyTorch、Megatron、DeepSpeed、CUDA、Ray
  领域标签　大模型训练、MLOps

设计约束：
  1. **只报告能定位到具体位置的结论**，不做语义猜测。
  2. **规则表公开**，每条命中都能对应到岗位要求的一条明示规则。
  3. **输入为空或格式不对时明确报错**，不静默返回空结果。

仅依赖 Python 标准库。

用法:
    python trainplat_jd193.py --help
    python trainplat_jd193.py --file input.txt
    python trainplat_jd193.py --text "示例内容"
    python trainplat_jd193.py --file input.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# 岗位规则表：每条规则独立可核查，命中即报，不做推断。
# 规则由岗位 JD 的技能线与领域标签派生——每条都能对回岗位要求。
# 目标岗位：华为 · 专家大模型训练平台工程师
RULES: list[dict] = [
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·缺失",
        "why": "岗位要求「PyTorch」在输入里找不到对应内容",
        "severity": "high",
        "keywords": ['missing', 'absent', '缺少', '没有', '未提供', 'PyTorch'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·冲突",
        "why": "两处关于「Megatron」的描述互相矛盾",
        "severity": "high",
        "keywords": ['conflict', 'contradict', '矛盾', '不一致', '相反', 'Megatron'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·越界",
        "why": "「DeepSpeed」的取值超出岗位场景的常规范围",
        "severity": "medium",
        "keywords": ['out of range', 'outlier', '越界', '超限', '超过', 'DeepSpeed'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·退化",
        "why": "「CUDA」相关的信息量不足以支撑判断",
        "severity": "medium",
        "keywords": ['degrade', 'insufficient', '不足', '太少', '无法', 'CUDA'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·泄漏",
        "why": "「Ray」相关的内容被意外暴露",
        "severity": "high",
        "keywords": ['leak', 'expose', '泄漏', '暴露', '明文', 'Ray'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·重复",
        "why": "同一处「大模型训练」信息被反复传递",
        "severity": "low",
        "keywords": ['duplicate', 'repeat', '重复', 'again', '大模型训练'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·陈旧",
        "why": "「MLOps」的时效性已过期",
        "severity": "medium",
        "keywords": ['stale', 'expired', '过期', '陈旧', '旧版', 'MLOps'],
    },
    {
        "name": "大模型训练平台工程师·分块策略对召回的影响·不可核",
        "why": "关于「大模型训练平台工程师」的说法无法追溯到具体证据",
        "severity": "high",
        "keywords": ['unverifiable', 'no evidence', '无法追溯', '没有来源', '大模型训练平台工程师'],
    },
]

# 定位规则：(名称, 正则, 说明)
LOCATOR_RULES: list[tuple[str, str, str]] = [('大模型训练平台工程师·分块策略对召回的影响·标注缺失', '\\b(todo|tbd|fixme|xxx)\\b', '占位符未替换'), ('大模型训练平台工程师·分块策略对召回的影响·硬编码', '(password|secret|api[_-]?key)\\s*[=:]\\s*[\\"\'][^\\"\']+', '疑似硬编码密钥'), ('大模型训练平台工程师·分块策略对召回的影响·调试残留', '(print\\(|console\\.log|breakpoint\\(\\))', '调试残留'), ('大模型训练平台工程师·分块策略对召回的影响·注释占位', '#\\s*(TODO|FIXME|待补|待定)\\b', '注释里留了待办')]



# 定位规则：(名称, 已编译正则, 说明)
# 正则在这里就编译好——如果留成字符串，每次分析都要重新 compile，
# 而 analyze 里直接调 rx.finditer 就会 AttributeError。
LOCATOR_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (name, re.compile(pat, re.I), why) for name, pat, why in LOCATOR_RULES
]


def trainplat_jd193_locate(text: str) -> list[dict]:
    """定位所有规则命中。"""
    out: list[dict] = []
    for name, rx, _ in LOCATOR_PATTERNS:
        for m in rx.finditer(text):
            out.append({
                "rule": name,
                "start": m.start(),
                "end": m.end(),
                "snippet": m.group(0)[:100],
            })
    out.sort(key=lambda d: d["start"])
    return out


def trainplat_jd193_analyze(text: str) -> dict:
    """主分析：违规位置与行号。"""
    hits = trainplat_jd193_locate(text)
    lines = text.splitlines()
    for h in hits:
        h["line"] = text.count("\n", 0, h["start"]) + 1
        h["line_text"] = lines[h["line"] - 1].strip()[:120] if h["line"] <= len(lines) else ""
    by_rule: dict[str, int] = {}
    for h in hits:
        by_rule[h["rule"]] = by_rule.get(h["rule"], 0) + 1
    return {
        "line_count": len(lines),
        "hits": hits,
        "hit_count": len(hits),
        "by_rule": by_rule,
        "clean": not hits,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render(result: dict) -> str:
    """把结果渲染成人读的文本。每个工具的版式不同。"""
    out: list[str] = []
    if not result.get("ok", True):
        out.append(f"无法分析：{{result.get('error', '未知原因')}}")
        return "\n".join(out)
    out.append(f"行数：{result['line_count']}  命中：{result['hit_count']}")
    out.append("")
    for h in result["hits"]:
        out.append(f"  第 {h['line']} 行  [{h['rule']}]  {h['snippet']!r}")
    if result["clean"]:
        out.append("没有命中任何规则。")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="trainplat_jd193",
        description="用一组明示的正则规则扫描输入，报告大模型训练平台工程师·分块策略对召回的影响相关的命中行与片段（仅依赖标准库）",
        epilog=(
            "示例:\n"
            "  python trainplat_jd193.py --file input.txt\n"
            "  python trainplat_jd193.py --text '示例内容'\n"
            "  python trainplat_jd193.py --file input.txt --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="输入文件路径")
    src.add_argument("--text", help="直接给出输入内容")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            print(f"读不了文件：{{e}}", file=sys.stderr)
            return 2
    else:
        content = args.text

    if not content.strip():
        print("输入是空的。", file=sys.stderr)
        return 2

    result = trainplat_jd193_analyze(content)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(render(result))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
