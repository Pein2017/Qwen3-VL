#!/usr/bin/env python3
"""
生成可疑工单的汇总报告和CSV文件
"""

import json
import csv
from pathlib import Path


def main():
    # 读取分析结果
    analysis_file = Path("/data/Qwen3-VL/analysis/duplicate_orders_analysis.json")
    suspicious_file = Path("/data/Qwen3-VL/analysis/suspicious_orders.json")

    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    with open(suspicious_file, "r", encoding="utf-8") as f:
        suspicious_data = json.load(f)

    # 生成汇总报告
    report_file = Path("/data/Qwen3-VL/analysis/suspicious_orders_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("可疑工单分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("判断标准：审核通过图片数 <= 2 且 审核不通过图片数 >= 3\n\n")

        f.write("汇总统计：\n")
        f.write("-" * 80 + "\n")
        total_duplicate = 0
        total_suspicious = 0

        for result in analysis_data:
            mission = result["mission"]
            duplicate = result["total_duplicate_orders"]
            suspicious = len(result["suspicious_orders"])
            total_duplicate += duplicate
            total_suspicious += suspicious

            f.write(f"{mission}:\n")
            f.write(f"  重复工单数: {duplicate}\n")
            f.write(f"  可疑工单数: {suspicious}\n")
            f.write(
                f"  可疑比例: {suspicious / duplicate * 100:.1f}%\n"
                if duplicate > 0
                else "  可疑比例: N/A\n"
            )
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("总计:\n")
        f.write(f"  总重复工单数: {total_duplicate}\n")
        f.write(f"  总可疑工单数: {total_suspicious}\n")
        f.write(
            f"  总可疑比例: {total_suspicious / total_duplicate * 100:.1f}%\n"
            if total_duplicate > 0
            else "  总可疑比例: N/A\n"
        )
        f.write("\n")

        # 按mission列出可疑工单统计
        f.write("=" * 80 + "\n")
        f.write("各Mission可疑工单详情（前20个）\n")
        f.write("=" * 80 + "\n\n")

        for mission, orders in suspicious_data["suspicious_orders_by_mission"].items():
            if not orders:
                continue

            f.write(f"{mission}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"共 {len(orders)} 个可疑工单\n\n")

            # 按审核不通过图片数排序，显示前20个
            sorted_orders = sorted(
                orders, key=lambda x: x["failed_images"], reverse=True
            )
            f.write("前20个最可疑的工单（按审核不通过图片数排序）：\n")
            for i, order in enumerate(sorted_orders[:20], 1):
                f.write(
                    f"  {i}. {order['order_name']}: "
                    f"审核通过={order['passed_images']}张, "
                    f"审核不通过={order['failed_images']}张, "
                    f"总计={order['total_images']}张\n"
                )
            f.write("\n")

    print(f"报告已生成: {report_file}")

    # 生成CSV文件
    csv_file = Path("/data/Qwen3-VL/analysis/suspicious_orders.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Mission",
                "工单名",
                "审核通过图片数",
                "审核不通过图片数",
                "总图片数",
                "审核通过目录",
                "审核不通过目录",
            ]
        )

        for mission, orders in suspicious_data["suspicious_orders_by_mission"].items():
            for order in orders:
                writer.writerow(
                    [
                        mission,
                        order["order_name"],
                        order["passed_images"],
                        order["failed_images"],
                        order["total_images"],
                        order["passed_dir"],
                        order["failed_dir"],
                    ]
                )

    print(f"CSV文件已生成: {csv_file}")
    print(f"\n共找到 {total_suspicious} 个可疑工单")


if __name__ == "__main__":
    main()
