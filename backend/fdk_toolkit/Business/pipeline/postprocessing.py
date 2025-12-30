from datetime import datetime, timedelta
import json
import os

def postprocessing(audit_response, result_id):

    business_reports_path = os.path.join(os.environ.get('REPORTS_FOLDER', './reports'), 'business')
    if not os.path.exists(business_reports_path):
        os.makedirs(business_reports_path)

    try:
        # Save report        
        report_filename = f"{result_id}_business_audit_report.json"
        report_path = os.path.join(business_reports_path, report_filename)
        with open(report_path, "w") as f:
            json.dump(audit_response, f, indent=2, default=str)
        
        # Generate business-specific summary
        summary_lines = build_business_summaries(audit_response)
        summary_text = " \n ".join(summary_lines)
        
        # Save summary
        summary_filename = f"{result_id}_business_audit_summary.txt"
        summary_path = os.path.join(business_reports_path, summary_filename)
        with open(summary_path, "w") as f:
            f.write(summary_text)

        return {"Success": report_filename + ", " + summary_filename}
        
    except Exception as e:
        print( f"Error: {str(e)}")
        return {"Error":f"Business audit failed."}
    
def build_business_summaries(audit: dict) -> list:
    """Business-specific human-readable summary"""
    lines = []
    
    # PROFESSIONAL SUMMARY
    lines.append("=== BUSINESS SERVICES PROFESSIONAL SUMMARY ===")
    lines.append("FDK Fairness Audit — Customer Equity & Service Interpretation")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Check for errors
    if "error" in audit:
        lines.append("❌ AUDIT ERROR DETECTED:")
        lines.append(f"   → Error: {audit['error']}")
        lines.append("   → The fairness audit could not complete due to technical issues.")
        lines.append("   → Please check your dataset format and try again.")
        lines.append("")
        return lines
    
    # DATASET OVERVIEW - STANDARDIZED ACROSS ALL DOMAINS
    lines.append("📊 DATASET OVERVIEW:")
    if "validation" in audit:
        validation_info = audit["validation"]
        lines.append(f"   → Total Customers Analyzed: {validation_info.get('sample_size', 'N/A')}")
        lines.append(f"   → Customer Segments: {validation_info.get('groups_analyzed', 'N/A')}")
        if 'statistical_power' in validation_info:
            lines.append(f"   → Statistical Power: {validation_info['statistical_power'].title()}")
    elif 'fairness_metrics' in audit and 'group_counts' in audit['fairness_metrics']:
        group_counts = audit['fairness_metrics']['group_counts']
        total_customers = sum(group_counts.values())
        num_groups = len(group_counts)
        lines.append(f"   → Total Customers Analyzed: {total_customers}")
        lines.append(f"   → Customer Segments: {num_groups}")
        if num_groups <= 10:
            lines.append(f"   → Segment Distribution: {dict(group_counts)}")
        else:
            lines.append(f"   → Largest Segment: {max(group_counts.values())} customers")
            lines.append(f"   → Smallest Segment: {min(group_counts.values())} customers")
    else:
        lines.append("   → Dataset statistics: Information not available")
    lines.append("")
    
    # Overall Assessment
    composite_score = audit.get("summary", {}).get("composite_bias_score")
    if composite_score is not None:
        lines.append("1) OVERALL CUSTOMER EQUITY ASSESSMENT:")
        lines.append(f"   → Composite Bias Score: {composite_score:.3f}")
        if composite_score > 0.10:
            lines.append("   → SEVERITY: HIGH - Significant customer equity concerns in service decisions")
            lines.append("   → ACTION: IMMEDIATE CUSTOMER EQUITY REVIEW REQUIRED")
        elif composite_score > 0.03:
            lines.append("   → SEVERITY: MEDIUM - Moderate customer equity concerns detected")
            lines.append("   → ACTION: SCHEDULE CUSTOMER EXPERIENCE REVIEW")
        else:
            lines.append("   → SEVERITY: LOW - Minimal customer equity concerns")
            lines.append("   → ACTION: CONTINUE MONITORING")
        lines.append("")
    
    # Key Business Metrics
    fairness_metrics = audit.get("fairness_metrics", {})
    
    if 'statistical_parity_difference' in fairness_metrics:
        spd = fairness_metrics['statistical_parity_difference']
        lines.append("2) SERVICE ALLOCATION DISPARITIES:")
        lines.append(f"   → Statistical Parity Difference: {spd:.3f}")
        if spd > 0.1:
            lines.append("     🚨 HIGH: Significant differences in service allocation across customer segments")
        elif spd > 0.05:
            lines.append("     ⚠️  MEDIUM: Noticeable service allocation variations")
        else:
            lines.append("     ✅ LOW: Consistent service allocation across customer segments")
        lines.append("")
    
    if 'fpr_difference' in fairness_metrics:
        fpr_diff = fairness_metrics['fpr_difference']
        lines.append("3) CUSTOMER ACCESS DISPARITIES:")
        lines.append(f"   → False Positive Rate Gap: {fpr_diff:.3f}")
        if fpr_diff > 0.1:
            lines.append("     🚨 HIGH: Some customer segments experience many more false service denials")
        elif fpr_diff > 0.05:
            lines.append("     ⚠️  MEDIUM: Moderate variation in false service denials")
        else:
            lines.append("     ✅ LOW: Consistent false positive rates across customer segments")
        lines.append("")
    
    # Business Recommendations
    lines.append("4) CUSTOMER EQUITY RECOMMENDATIONS:")
    if composite_score and composite_score > 0.10:
        lines.append("   🚨 IMMEDIATE EQUITY ACTIONS REQUIRED:")
        lines.append("   • Conduct comprehensive customer equity investigation")
        lines.append("   • Review service allocation decision-making processes")
        lines.append("   • Implement customer equity mitigation protocols")
        lines.append("   • Consider external customer experience audit")
    elif composite_score and composite_score > 0.03:
        lines.append("   ⚖️  RECOMMENDED CUSTOMER REVIEW:")
        lines.append("   • Schedule systematic customer equity review")
        lines.append("   • Monitor service allocation patterns by customer segment")
        lines.append("   • Document customer equity considerations")
        lines.append("   • Plan procedural improvements for equity")
    else:
        lines.append("   ✅ CUSTOMER EQUITY STANDARDS MAINTAINED:")
        lines.append("   • Continue regular customer equity monitoring")
        lines.append("   • Maintain current customer equity standards")
        lines.append("   • Document customer equity assessment")
    lines.append("")
    
    # PUBLIC SUMMARY
    lines.append("=== CUSTOMER TRANSPARENCY SUMMARY ===")
    lines.append("Plain-English Interpretation for Customer Trust:")
    lines.append("")
    
    # Check for high individual metrics even if composite score is low
    high_bias_detected = False
    medium_bias_detected = False
    
    # Check specific high-impact metrics
    if 'statistical_parity_difference' in fairness_metrics and fairness_metrics['statistical_parity_difference'] > 0.1:
        high_bias_detected = True
    if 'equal_opportunity_difference' in fairness_metrics and fairness_metrics['equal_opportunity_difference'] > 0.1:
        high_bias_detected = True
    if 'average_odds_difference' in fairness_metrics and fairness_metrics['average_odds_difference'] > 0.1:
        high_bias_detected = True
    
    # Check for medium bias indicators
    if not high_bias_detected:
        if 'statistical_parity_difference' in fairness_metrics and fairness_metrics['statistical_parity_difference'] > 0.05:
            medium_bias_detected = True
        if 'equal_opportunity_difference' in fairness_metrics and fairness_metrics['equal_opportunity_difference'] > 0.05:
            medium_bias_detected = True
    
    # Determine public summary based on actual bias levels
    if high_bias_detected or (composite_score and composite_score > 0.10):
        lines.append("🔴 SIGNIFICANT EQUITY CONCERNS")
        lines.append("")
        lines.append("This business tool shows substantial differences in how it treats different customer segments.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Service decisions may be inconsistent across customer groups")
        lines.append("• Some segments may experience different service access rates")
        lines.append("• Additional review of business processes is recommended")
    elif medium_bias_detected or (composite_score and composite_score > 0.03):
        lines.append("🟡 MODERATE EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This business tool generally works fairly but shows some variation across customer segments.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• The tool is mostly consistent in its business decisions")
        lines.append("• Some small differences in treatment may exist")
        lines.append("• Ongoing customer equity monitoring is recommended")
    else:
        lines.append("🟢 GOOD EQUITY ASSESSMENT")
        lines.append("")
        lines.append("This business tool demonstrates consistent treatment across all customer segments.")
        lines.append("")
        lines.append("What this means:")
        lines.append("• Business decisions are applied consistently regardless of customer background")
        lines.append("• The tool meets customer equity standards")
        lines.append("• Treatment is equitable across different customer segments")
    
    lines.append("")
    
    # CUSTOMER EQUITY DISCLAIMER
    lines.append("=== CUSTOMER EQUITY DISCLAIMER ===")
    lines.append("This customer equity audit complies with:")
    lines.append("• Consumer protection laws")
    lines.append("• Fair business practice regulations")
    lines.append("• Anti-discrimination business laws")
    lines.append("• Algorithmic accountability frameworks in business services")
    lines.append("")
    lines.append("BUSINESS NOTICE: This tool is for customer equity assessment only and does not:")
    lines.append("• Provide business guarantees or outcomes")
    lines.append("• Determine customer eligibility")
    lines.append("• Replace professional business consultation")
    lines.append("")
    lines.append("For customer equity concerns, consult qualified business professionals.")
    
    return lines
