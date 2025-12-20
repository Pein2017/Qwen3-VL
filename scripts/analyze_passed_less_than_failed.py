#!/usr/bin/env python3
"""
分析审核通过图片数少于审核不通过图片数的工单，查看数据分布
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

def analyze_all_missions():
    """分析所有mission目录"""
    all_results = []
    
    # 读取BBU missions的分析结果
    bbu_analysis_file = Path("/data/Qwen3-VL/analysis/duplicate_orders_analysis.json")
    if bbu_analysis_file.exists():
        with open(bbu_analysis_file, 'r', encoding='utf-8') as f:
            bbu_data = json.load(f)
        
        for mission_data in bbu_data:
            # 找出审核通过图片数 < 审核不通过图片数的工单
            passed_less_than_failed = [
                order for order in mission_data['all_duplicate_orders']
                if order['passed_images'] < order['failed_images']
            ]
            
            result = {
                'mission': mission_data['mission'],
                'mission_path': f"group_data/bbu_scene_2.0_order/{mission_data['mission']}",
                'total_duplicate_orders': mission_data['total_duplicate_orders'],
                'passed_less_than_failed': passed_less_than_failed,
                'count': len(passed_less_than_failed),
            }
            all_results.append(result)
            
            print(f"\n{result['mission']}:")
            print(f"  重复工单数: {result['total_duplicate_orders']}")
            print(f"  审核通过 < 审核不通过: {result['count']} 个")
            if result['total_duplicate_orders'] > 0:
                print(f"  占比: {result['count']/result['total_duplicate_orders']*100:.1f}%")
    
    # 读取RRU mission的分析结果
    rru_analysis_file = Path("/data/Qwen3-VL/analysis/RRU安装检查_analysis.json")
    if rru_analysis_file.exists():
        with open(rru_analysis_file, 'r', encoding='utf-8') as f:
            rru_data = json.load(f)
        
        # 找出审核通过图片数 < 审核不通过图片数的工单
        passed_less_than_failed = [
            order for order in rru_data['all_duplicate_orders']
            if order['passed_images'] < order['failed_images']
        ]
        
        result = {
            'mission': rru_data['mission'],
            'mission_path': rru_data.get('mission_path', 'group_data/rru_scene/rru_scene/rru_scene/RRU安装检查'),
            'total_duplicate_orders': rru_data['total_duplicate_orders'],
            'passed_less_than_failed': passed_less_than_failed,
            'count': len(passed_less_than_failed),
        }
        all_results.append(result)
        
        print(f"\n{result['mission']}:")
        print(f"  重复工单数: {result['total_duplicate_orders']}")
        print(f"  审核通过 < 审核不通过: {result['count']} 个")
        if result['total_duplicate_orders'] > 0:
            print(f"  占比: {result['count']/result['total_duplicate_orders']*100:.1f}%")
    
    return all_results


def generate_distribution_report(all_results):
    """生成数据分布报告"""
    report_file = Path("/data/Qwen3-VL/analysis/passed_less_than_failed_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("审核通过图片数 < 审核不通过图片数的工单分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        total_duplicate = 0
        total_passed_less = 0
        
        for result in all_results:
            mission = result['mission']
            duplicate = result['total_duplicate_orders']
            passed_less = result['count']
            total_duplicate += duplicate
            total_passed_less += passed_less
            
            f.write(f"{mission}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  重复工单数: {duplicate}\n")
            f.write(f"  审核通过 < 审核不通过: {passed_less} 个\n")
            if duplicate > 0:
                f.write(f"  占比: {passed_less/duplicate*100:.1f}%\n")
            f.write("\n")
            
            # 统计分布
            if passed_less > 0:
                f.write("  数据分布统计:\n")
                # 按差值分组统计
                diff_groups = defaultdict(int)
                ratio_groups = defaultdict(int)
                
                for order in result['passed_less_than_failed']:
                    diff = order['failed_images'] - order['passed_images']
                    diff_groups[diff] += 1
                    
                    if order['passed_images'] > 0:
                        ratio = order['failed_images'] / order['passed_images']
                        if ratio < 2:
                            ratio_groups['1-2倍'] += 1
                        elif ratio < 3:
                            ratio_groups['2-3倍'] += 1
                        elif ratio < 5:
                            ratio_groups['3-5倍'] += 1
                        else:
                            ratio_groups['5倍以上'] += 1
                    else:
                        ratio_groups['审核通过=0'] += 1
                
                f.write("    按差值分布（审核不通过 - 审核通过）:\n")
                for diff in sorted(diff_groups.keys()):
                    f.write(f"      差值={diff}张: {diff_groups[diff]} 个工单\n")
                
                f.write("    按倍数分布（审核不通过 / 审核通过）:\n")
                for ratio_range in ['1-2倍', '2-3倍', '3-5倍', '5倍以上', '审核通过=0']:
                    if ratio_range in ratio_groups:
                        f.write(f"      {ratio_range}: {ratio_groups[ratio_range]} 个工单\n")
                
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("总计:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  总重复工单数: {total_duplicate}\n")
        f.write(f"  审核通过 < 审核不通过: {total_passed_less} 个\n")
        if total_duplicate > 0:
            f.write(f"  总占比: {total_passed_less/total_duplicate*100:.1f}%\n")
    
    print(f"\n报告已生成: {report_file}")


def generate_csv(all_results):
    """生成CSV文件"""
    csv_file = Path("/data/Qwen3-VL/analysis/passed_less_than_failed.csv")
    
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mission', '工单名', '审核通过图片数', '审核不通过图片数', '总图片数', 
                        '差值(不通过-通过)', '倍数(不通过/通过)', 
                        '审核通过目录', '审核不通过目录'])
        
        for result in all_results:
            for order in result['passed_less_than_failed']:
                diff = order['failed_images'] - order['passed_images']
                ratio = order['failed_images'] / order['passed_images'] if order['passed_images'] > 0 else float('inf')
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                
                writer.writerow([
                    result['mission'],
                    order['order_name'],
                    order['passed_images'],
                    order['failed_images'],
                    order['total_images'],
                    diff,
                    ratio_str,
                    order['passed_dir'],
                    order['failed_dir']
                ])
    
    print(f"CSV文件已生成: {csv_file}")


def main():
    print("开始分析所有mission...")
    print("从已有分析结果中提取数据...")
    
    # 使用已有的分析结果
    all_results = analyze_all_missions()
    
    if not all_results:
        print("[ERROR] 没有找到任何分析结果")
        return
    
    # 生成报告
    generate_distribution_report(all_results)
    generate_csv(all_results)
    
    # 打印汇总
    total_duplicate = sum(r['total_duplicate_orders'] for r in all_results)
    total_passed_less = sum(r['count'] for r in all_results)
    
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"总重复工单数: {total_duplicate}")
    print(f"审核通过 < 审核不通过: {total_passed_less} 个")
    if total_duplicate > 0:
        print(f"占比: {total_passed_less/total_duplicate*100:.1f}%")


if __name__ == "__main__":
    main()

