#!/usr/bin/env python3
"""
分析同时出现在"审核通过"和"审核不通过"中的工单，并统计图片数量。
找出疑似有问题的工单（审核通过图片很少，但审核不通过图片很多）。
"""
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}


def count_images(directory: Path) -> int:
    """统计目录中的图片数量（递归）"""
    if not directory.exists():
        return 0
    count = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def analyze_mission(mission_dir: Path) -> Dict:
    """分析单个mission目录"""
    passed_dir = mission_dir / "审核通过"
    failed_dir = mission_dir / "审核不通过"
    
    if not passed_dir.exists() or not failed_dir.exists():
        print(f"[WARNING] 目录不存在: {mission_dir}")
        return {}
    
    # 获取所有工单名
    passed_orders = {d.name for d in passed_dir.iterdir() if d.is_dir()}
    failed_orders = {d.name for d in failed_dir.iterdir() if d.is_dir()}
    
    # 找出同时出现在两个目录中的工单
    duplicate_orders = passed_orders & failed_orders
    
    print(f"\n{'='*80}")
    print(f"Mission: {mission_dir.name}")
    print(f"{'='*80}")
    print(f"审核通过工单数: {len(passed_orders)}")
    print(f"审核不通过工单数: {len(failed_orders)}")
    print(f"重复工单数: {len(duplicate_orders)}")
    
    results = []
    suspicious_orders = []
    
    for order_name in sorted(duplicate_orders):
        passed_order_dir = passed_dir / order_name
        failed_order_dir = failed_dir / order_name
        
        passed_count = count_images(passed_order_dir)
        failed_count = count_images(failed_order_dir)
        total_count = passed_count + failed_count
        
        result = {
            'order_name': order_name,
            'passed_images': passed_count,
            'failed_images': failed_count,
            'total_images': total_count,
            'passed_dir': str(passed_order_dir),
            'failed_dir': str(failed_order_dir),
        }
        results.append(result)
        
        # 判断是否可疑：审核通过图片很少（<=2），但审核不通过图片很多（>=3）
        if passed_count <= 2 and failed_count >= 3:
            suspicious_orders.append(result)
            print(f"\n[可疑] {order_name}: 审核通过={passed_count}张, 审核不通过={failed_count}张")
    
    return {
        'mission': mission_dir.name,
        'total_duplicate_orders': len(duplicate_orders),
        'suspicious_orders': suspicious_orders,
        'all_duplicate_orders': results,
    }


def main():
    base_dir = Path("/data/Qwen3-VL/group_data/bbu_scene_2.0_order")
    
    missions = [
        "BBU安装方式检查（正装）",
        "BBU接地线检查",
        "BBU线缆布放要求",
        "挡风板安装检查",
    ]
    
    all_results = []
    
    for mission_name in missions:
        mission_dir = base_dir / mission_name
        if not mission_dir.exists():
            print(f"[ERROR] Mission目录不存在: {mission_dir}")
            continue
        
        result = analyze_mission(mission_dir)
        if result:
            all_results.append(result)
    
    # 生成汇总报告
    print(f"\n\n{'='*80}")
    print("汇总报告")
    print(f"{'='*80}")
    
    total_suspicious = 0
    total_duplicate = 0
    
    for result in all_results:
        print(f"\n{result['mission']}:")
        print(f"  重复工单数: {result['total_duplicate_orders']}")
        print(f"  可疑工单数: {len(result['suspicious_orders'])}")
        total_duplicate += result['total_duplicate_orders']
        total_suspicious += len(result['suspicious_orders'])
    
    print(f"\n总计:")
    print(f"  总重复工单数: {total_duplicate}")
    print(f"  总可疑工单数: {total_suspicious}")
    
    # 保存详细结果到JSON文件
    output_file = Path("/data/Qwen3-VL/analysis/duplicate_orders_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 保存可疑工单列表到单独文件
    suspicious_file = Path("/data/Qwen3-VL/analysis/suspicious_orders.json")
    suspicious_data = {
        'total_suspicious': total_suspicious,
        'suspicious_orders_by_mission': {
            r['mission']: r['suspicious_orders'] for r in all_results
        }
    }
    
    with open(suspicious_file, 'w', encoding='utf-8') as f:
        json.dump(suspicious_data, f, ensure_ascii=False, indent=2)
    
    print(f"可疑工单列表已保存到: {suspicious_file}")
    
    # 打印所有可疑工单的详细信息
    if total_suspicious > 0:
        print(f"\n\n{'='*80}")
        print("所有可疑工单详情")
        print(f"{'='*80}")
        for result in all_results:
            if result['suspicious_orders']:
                print(f"\n{result['mission']}:")
                for order in result['suspicious_orders']:
                    print(f"  {order['order_name']}: 审核通过={order['passed_images']}张, "
                          f"审核不通过={order['failed_images']}张, 总计={order['total_images']}张")


if __name__ == "__main__":
    main()

