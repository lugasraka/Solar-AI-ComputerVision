"""
SolarVision AI - Business Impact Calculator
Calculates ROI and cost savings for solar farm operators
"""

import numpy as np
from datetime import datetime

class BusinessCalculator:
    """Calculate business value and ROI for SolarVision AI deployment"""
    
    def __init__(self):
        # Cost assumptions
        self.manual_cost_per_panel = 1.50  # USD
        self.ai_cost_per_panel = 0.20      # USD
        self.average_panel_wattage = 330   # Watts
        self.panels_per_mw = 1000 / (self.average_panel_wattage / 1000)  # ~3030 panels per MW
        
        # Inspection frequency
        self.manual_inspections_per_year = 4   # Quarterly
        self.ai_inspections_per_year = 12      # Monthly
        
        # Performance improvements
        self.detection_accuracy_improvement = 0.15  # 15% better detection
        self.time_reduction_factor = 10  # 10x faster
        
    def calculate(self, farm_size_mw):
        """
        Calculate business metrics for a solar farm
        
        Args:
            farm_size_mw: Size of solar farm in megawatts
            
        Returns:
            dict with all business metrics
        """
        # Calculate panel count
        panel_count = int(farm_size_mw * self.panels_per_mw)
        
        # Annual costs
        manual_annual_cost = panel_count * self.manual_cost_per_panel * self.manual_inspections_per_year
        ai_annual_cost = panel_count * self.ai_cost_per_panel * self.ai_inspections_per_year
        
        # Savings
        annual_savings = manual_annual_cost - ai_annual_cost
        cost_reduction_pct = (annual_savings / manual_annual_cost) * 100
        
        # Time savings
        manual_hours_per_inspection = panel_count / 500  # 500 panels per day
        ai_hours_per_inspection = panel_count / 5000     # 5000 panels per day
        time_saved_per_inspection = manual_hours_per_inspection - ai_hours_per_inspection
        total_time_saved_annual = time_saved_per_inspection * self.ai_inspections_per_year
        
        # Efficiency gains
        additional_energy_yield = farm_size_mw * 1000 * self.detection_accuracy_improvement  # kWh
        energy_value = additional_energy_yield * 0.10  # Assuming $0.10 per kWh
        
        # Payback period (if any implementation costs)
        implementation_cost = 50000  # One-time setup cost estimate
        payback_months = (implementation_cost / (annual_savings + energy_value)) * 12
        
        return {
            'farm_size_mw': farm_size_mw,
            'panel_count': panel_count,
            'manual_annual_cost': manual_annual_cost,
            'ai_annual_cost': ai_annual_cost,
            'annual_savings': annual_savings,
            'cost_reduction_pct': cost_reduction_pct,
            'time_saved_hours': total_time_saved_annual,
            'additional_energy_yield_kwh': additional_energy_yield,
            'energy_value': energy_value,
            'total_annual_benefit': annual_savings + energy_value,
            'inspection_frequency_increase': f"{self.manual_inspections_per_year}x → {self.ai_inspections_per_year}x",
            'payback_months': max(0, payback_months),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def generate_report_text(self, metrics):
        """Generate human-readable business report"""
        report = []
        report.append("=" * 60)
        report.append("SOLARVISION AI - BUSINESS IMPACT REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Generated: {metrics['generated_at']}")
        report.append("")
        report.append("FARM SPECIFICATIONS:")
        report.append(f"  Solar Farm Size: {metrics['farm_size_mw']:.1f} MW")
        report.append(f"  Total Panels: {metrics['panel_count']:,}")
        report.append("")
        report.append("COST ANALYSIS:")
        report.append(f"  Manual Inspection (Annual): ${metrics['manual_annual_cost']:,.2f}")
        report.append(f"  AI-Powered Inspection (Annual): ${metrics['ai_annual_cost']:,.2f}")
        report.append(f"  Annual Savings: ${metrics['annual_savings']:,.2f}")
        report.append(f"  Cost Reduction: {metrics['cost_reduction_pct']:.1f}%")
        report.append("")
        report.append("EFFICIENCY GAINS:")
        report.append(f"  Time Saved (Annual): {metrics['time_saved_hours']:,.0f} hours")
        report.append(f"  Inspection Frequency: {metrics['inspection_frequency_increase']}")
        report.append(f"  Detection Accuracy Improvement: +15%")
        report.append("")
        report.append("ENERGY IMPACT:")
        report.append(f"  Additional Energy Yield: {metrics['additional_energy_yield_kwh']:,.0f} kWh/year")
        report.append(f"  Energy Value: ${metrics['energy_value']:,.2f}/year")
        report.append("")
        report.append("ROI SUMMARY:")
        report.append(f"  Total Annual Benefit: ${metrics['total_annual_benefit']:,.2f}")
        if metrics['payback_months'] > 0:
            report.append(f"  Payback Period: {metrics['payback_months']:.1f} months")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_summary_chart_data(self, metrics):
        """Get data for visualization charts"""
        return {
            'cost_comparison': {
                'labels': ['Manual Inspection', 'AI-Powered'],
                'values': [metrics['manual_annual_cost'], metrics['ai_annual_cost']]
            },
            'benefits_breakdown': {
                'labels': ['Cost Savings', 'Energy Value'],
                'values': [metrics['annual_savings'], metrics['energy_value']]
            }
        }


if __name__ == '__main__':
    # Test calculator
    calc = BusinessCalculator()
    metrics = calc.calculate(100)  # 100 MW farm
    print(calc.generate_report_text(metrics))
