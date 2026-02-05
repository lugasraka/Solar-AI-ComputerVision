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
        
        # Efficiency gains - realistic calculation based on defect detection
        # Industry data: ~8% of panels have defects, avg 20% power loss per defect
        typical_defect_rate = 0.08  # 8% of panels have defects
        avg_power_loss_per_defect = 0.20  # 20% power loss per defective panel
        peak_sun_hours_per_day = 5  # Average peak sun hours
        
        # Calculate energy saved through early defect detection
        panels_with_defects = panel_count * typical_defect_rate
        power_loss_per_panel_kw = (self.average_panel_wattage / 1000) * avg_power_loss_per_defect
        total_power_loss_mw = (panels_with_defects * power_loss_per_panel_kw) / 1000
        
        # Annual energy loss from defects (if undetected)
        annual_energy_loss_mwh = total_power_loss_mw * peak_sun_hours_per_day * 365
        
        # 15% better detection = finding 15% more defects 3 months earlier (quarterly→monthly)
        # Each month earlier detection saves 1/12 of annual loss for those panels
        months_saved = 3  # Monthly vs quarterly detection
        detection_improvement = 0.15  # 15% better detection rate
        energy_saved_mwh = annual_energy_loss_mwh * detection_improvement * (months_saved / 12)
        energy_saved_kwh = energy_saved_mwh * 1000
        
        energy_value = energy_saved_kwh * 0.10  # $0.10 per kWh
        
        # Store calculation details for transparency
        energy_calculation_details = {
            'panels_with_defects': int(panels_with_defects),
            'total_power_loss_mw': total_power_loss_mw,
            'annual_energy_loss_mwh': annual_energy_loss_mwh,
            'energy_saved_mwh': energy_saved_mwh,
            'detection_improvement': detection_improvement,
            'months_saved': months_saved
        }
        
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
            'additional_energy_yield_kwh': energy_saved_kwh,
            'energy_value': energy_value,
            'total_annual_benefit': annual_savings + energy_value,
            'inspection_frequency_increase': f"{self.manual_inspections_per_year}x → {self.ai_inspections_per_year}x",
            'payback_months': max(0, payback_months),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'energy_calculation_details': energy_calculation_details
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
    
    def format_currency(self, amount):
        """Format amount as currency with K/M suffix for readability"""
        if amount >= 1_000_000:
            return f"${amount/1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"${amount/1_000:.0f}K"
        else:
            return f"${amount:,.0f}"
    
    def format_number(self, num):
        """Format large numbers with K/M suffix"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.0f}K"
        else:
            return f"{num:,.0f}"
    
    def hours_to_days(self, hours):
        """Convert hours to working days (8 hours/day)"""
        days = hours / 8
        if days >= 365:
            return f"{days/365:.1f} years"
        elif days >= 30:
            return f"{days/30:.0f} months"
        else:
            return f"{days:.0f} days"
    
    def get_kpi_cards(self, metrics):
        """Generate KPI card data for display"""
        return {
            'annual_savings': {
                'value': self.format_currency(metrics['annual_savings']),
                'raw_value': metrics['annual_savings'],
                'label': 'Annual Savings',
                'icon': '💰',
                'color': '#27ae60',
                'subtitle': f"{metrics['cost_reduction_pct']:.1f}% cost reduction"
            },
            'time_saved': {
                'value': self.format_number(metrics['time_saved_hours']),
                'raw_value': metrics['time_saved_hours'],
                'label': 'Hours Saved',
                'icon': '⏱️',
                'color': '#3498db',
                'subtitle': f"≈ {self.hours_to_days(metrics['time_saved_hours'])} of work"
            },
            'payback_period': {
                'value': f"{metrics['payback_months']:.1f}",
                'raw_value': metrics['payback_months'],
                'label': 'Payback Period',
                'icon': '🎯',
                'color': '#9b59b6',
                'subtitle': 'months to break even'
            },
            'energy_value': {
                'value': self.format_currency(metrics['energy_value']),
                'raw_value': metrics['energy_value'],
                'label': 'Energy Value',
                'icon': '⚡',
                'color': '#f39c12',
                'subtitle': f"{self.format_number(metrics['additional_energy_yield_kwh'])} kWh/year"
            },
            'panel_count': {
                'value': self.format_number(metrics['panel_count']),
                'raw_value': metrics['panel_count'],
                'label': 'Panels Protected',
                'icon': '☀️',
                'color': '#e74c3c',
                'subtitle': f"{metrics['farm_size_mw']:.1f} MW farm"
            },
            'total_benefit': {
                'value': self.format_currency(metrics['total_annual_benefit']),
                'raw_value': metrics['total_annual_benefit'],
                'label': 'Total Annual Benefit',
                'icon': '📈',
                'color': '#1abc9c',
                'subtitle': 'savings + energy value'
            }
        }
    
    def generate_kpi_summary(self, metrics):
        """Generate simplified summary for display"""
        kpis = self.get_kpi_cards(metrics)
        
        summary = []
        summary.append(f"☀️ {metrics['farm_size_mw']:.1f} MW Solar Farm Analysis")
        summary.append(f"📊 {self.format_number(metrics['panel_count'])} panels | {metrics['inspection_frequency_increase']} inspections/year")
        summary.append("")
        summary.append("🎯 KEY HIGHLIGHTS:")
        summary.append(f"   • Save {self.format_currency(metrics['annual_savings'])} annually ({metrics['cost_reduction_pct']:.0f}% reduction)")
        summary.append(f"   • Free up {self.hours_to_days(metrics['time_saved_hours'])} of labor")
        summary.append(f"   • ROI payback in {metrics['payback_months']:.1f} months")
        summary.append("")
        summary.append("💡 WHAT THIS MEANS:")
        summary.append("   • 3x more frequent inspections (quarterly → monthly)")
        summary.append("   • 15% better defect detection accuracy")
        summary.append("   • Reduced safety risks from electrical failures")
        
        return "\n".join(summary)


if __name__ == '__main__':
    # Test calculator
    calc = BusinessCalculator()
    metrics = calc.calculate(100)  # 100 MW farm
    print(calc.generate_report_text(metrics))
