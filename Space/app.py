"""
SolarVision AI - Gradio Demo Application
Interactive web interface for PV panel defect detection with Grad-CAM support
"""

import os
import sys
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
import base64

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from inference import get_predictor, GRADCAM_AVAILABLE
from business_calculator import BusinessCalculator
from report_generator import PDFReportGenerator
from utils import AutoShutdownTimer, ProgressTracker, format_confidence, get_confidence_color

# Configuration
APP_TITLE = "☀️ SolarVision AI"
APP_DESCRIPTION = """
# SolarVision AI - PV Panel Defect Detection

**Automated defect detection for solar panel installations using AI**

Upload images of solar panels to detect defects including:
- Bird droppings
- Dust accumulation  
- Electrical damage
- Physical damage
- Snow coverage

**Models Available:**
- **SVM Mode**: ResNet18 + SVM (96.8% accuracy) - Higher accuracy
- **CNN Mode**: End-to-End ResNet18 (95.8% accuracy) - With Grad-CAM explainability

*Auto-closes after 30 minutes of inactivity*
"""

# Initialize components
predictor = None
calculator = BusinessCalculator()
report_gen = PDFReportGenerator()
timer = AutoShutdownTimer(timeout_minutes=30, warning_minutes=25)

# Global state
processed_results = []

def load_predictor():
    """Lazy load predictor"""
    global predictor
    if predictor is None:
        models_dir = Path(__file__).parent.parent / 'models'
        predictor = get_predictor(str(models_dir))
    return predictor


def predict_single(image, model_choice):
    """Process single image with selected model"""
    global processed_results
    
    # Reset timer on activity
    timer.reset()
    
    if image is None:
        return None, "Please upload an image", "", "", None, None, None
    
    try:
        # Handle different image input types from Gradio
        if isinstance(image, str):
            # Image is already a file path
            temp_path = Path(image)
            display_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Image is a numpy array, save it
            temp_path = Path(tempfile.gettempdir()) / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            Image.fromarray(image).save(temp_path)
            display_image = image
        elif hasattr(image, 'save'):
            # Image is a PIL Image object
            temp_path = Path(tempfile.gettempdir()) / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image.save(temp_path)
            display_image = image
        else:
            return None, f"Unsupported image type: {type(image)}", "", "", None, None, None
        
        # Determine model to use
        use_cnn = (model_choice == "CNN with Grad-CAM (95.8%)")
        
        # Get prediction
        if use_cnn and GRADCAM_AVAILABLE:
            # Use CNN with Grad-CAM
            pred, gradcam_images = load_predictor().predict_with_gradcam(str(temp_path))
            
            # Extract Grad-CAM images
            gradcam_heatmap = gradcam_images['heatmap'] if gradcam_images else None
            gradcam_overlay = gradcam_images['overlay'] if gradcam_images else None
            gradcam_original = gradcam_images['original'] if gradcam_images else None
        else:
            # Use SVM or CNN without Grad-CAM
            pred = load_predictor().predict(str(temp_path), use_cnn=use_cnn)
            gradcam_heatmap = None
            gradcam_overlay = None
            gradcam_original = None
        
        processed_results = [pred]
        
        # Create results display
        gradcam_status = ""
        if use_cnn:
            if GRADCAM_AVAILABLE:
                gradcam_status = f"<p style='font-size: 12px; color: #3498db;'>🔍 Grad-CAM visualization enabled</p>"
            else:
                gradcam_status = f"<p style='font-size: 12px; color: #e74c3c;'>⚠️ Grad-CAM not available (install grad-cam)</p>"
        
        result_html = f"""
        <div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa;'>
            <h3 style='color: {get_confidence_color(pred["confidence"])};'>
                Predicted: {pred['predicted_class']}
            </h3>
            <p style='font-size: 18px;'>
                Confidence: <strong>{format_confidence(pred['confidence'])}</strong>
            </p>
            <p style='font-size: 14px; color: #7f8c8d;'>
                Model: {pred['model_used']}
            </p>
            {gradcam_status}
        </div>
        """
        
        # Create confidence chart
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = list(pred['all_probabilities'].keys())
        probs = list(pred['all_probabilities'].values())
        
        colors_list = ['#27ae60' if c == pred['predicted_class'] else '#95a5a6' for c in classes]
        bars = ax.barh(classes, probs, color=colors_list)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.1%}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Clean up temp file only if we created it
        if not isinstance(image, str):
            temp_path.unlink(missing_ok=True)
        
        return (display_image, result_html, fig, pred['filename'], 
                gradcam_original, gradcam_heatmap, gradcam_overlay)
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return image, f"Error: {str(e)}", None, "", None, None, None


def predict_batch_zip(zip_file, model_choice, progress=gr.Progress()):
    """Process batch of images from ZIP file"""
    global processed_results
    
    # Reset timer
    timer.reset()
    
    if zip_file is None:
        return pd.DataFrame(), "Please upload a ZIP file", "No results yet"
    
    try:
        # Extract ZIP
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(temp_dir.rglob(f'*{ext}'))
            image_files.extend(temp_dir.rglob(f'*{ext.upper()}'))
        
        if not image_files:
            return pd.DataFrame(), "No images found in ZIP file", "No results"
        
        # Determine model to use
        use_cnn = (model_choice == "CNN with Grad-CAM (95.8%)")
        
        # Process images
        results = []
        tracker = ProgressTracker(len(image_files))
        
        for i, img_path in enumerate(image_files):
            try:
                pred = load_predictor().predict(str(img_path), use_cnn=use_cnn)
                results.append(pred)
                
                # Update progress
                tracker.update()
                progress(tracker.current / tracker.total_items, 
                        desc=f"Processing {tracker.current}/{tracker.total_items}")
                        
            except Exception as e:
                results.append({'filename': img_path.name, 'error': str(e)})
        
        processed_results = results
        
        # Create results dataframe
        df_data = []
        for r in results:
            if 'error' not in r:
                df_data.append({
                    'Filename': r['filename'],
                    'Prediction': r['predicted_class'],
                    'Confidence': f"{r['confidence']:.1%}",
                    'Top Class': r['top3'][0][0] if r['top3'] else 'N/A',
                    'Model': 'CNN' if use_cnn else 'SVM'
                })
        
        df = pd.DataFrame(df_data)
        
        # Summary text
        model_str = "CNN" if use_cnn else "SVM"
        summary = f"Processed {len(results)} images using {model_str}"
        if any('error' in r for r in results):
            errors = sum(1 for r in results if 'error' in r)
            summary += f" ({errors} errors)"
        
        return df, summary, "Ready to export"
        
    except Exception as e:
        import traceback
        print(f"Batch processing error: {traceback.format_exc()}")
        return pd.DataFrame(), f"Error processing ZIP: {str(e)}", "Error"


def update_model_info(model_choice):
    """Update model information display"""
    if model_choice == "SVM (96.8% accuracy)":
        return """
        **SVM Mode (ResNet18 + SVM)**
        - Accuracy: 96.84%
        - Grad-CAM: Not available
        - Best for: Maximum accuracy
        
        Uses ResNet18 for feature extraction and SVM for classification.
        """
    else:
        gradcam_status = "✅ Available" if GRADCAM_AVAILABLE else "❌ Not installed"
        return f"""
        **CNN Mode (End-to-End ResNet18)**
        - Accuracy: 95.79%
        - Grad-CAM: {gradcam_status}
        - Best for: Explainable AI with visual attention maps
        
        Full CNN model that generates Grad-CAM visualizations showing 
        which regions the model focused on for its prediction.
        """


def export_csv():
    """Export results to CSV"""
    timer.reset()
    
    if not processed_results:
        return gr.update(value=None, visible=False), "No results to export. Please process images first."
    
    try:
        csv_path = report_gen.generate_csv(processed_results)
        if csv_path and os.path.exists(csv_path):
            return gr.update(value=csv_path, visible=True), f"✅ CSV ready: {Path(csv_path).name}"
        else:
            return gr.update(value=None, visible=False), "❌ Failed to generate CSV"
    except Exception as e:
        return gr.update(value=None, visible=False), f"❌ Error: {str(e)}"


def export_pdf():
    """Export results to PDF"""
    timer.reset()
    
    if not processed_results:
        return gr.update(value=None, visible=False), "No results to export. Please process images first."
    
    try:
        pdf_path = report_gen.generate_report(processed_results)
        if pdf_path and os.path.exists(pdf_path):
            return gr.update(value=pdf_path, visible=True), f"✅ PDF ready: {Path(pdf_path).name}"
        else:
            return gr.update(value=None, visible=False), "❌ Failed to generate PDF"
    except Exception as e:
        return gr.update(value=None, visible=False), f"❌ Error: {str(e)}"


def calculate_business(farm_size):
    """Calculate business impact with KPI cards"""
    timer.reset()
    
    try:
        metrics = calculator.calculate(farm_size)
        kpis = calculator.get_kpi_cards(metrics)
        
        # Generate KPI cards HTML
        kpi_html = generate_kpi_cards_html(kpis)
        
        # Generate summary text
        summary_text = calculator.generate_kpi_summary(metrics)
        
        # Create enhanced visualization
        fig = create_business_charts(metrics)
        
        return kpi_html, summary_text, fig
        
    except Exception as e:
        import traceback
        print(f"Error in calculate_business: {traceback.format_exc()}")
        return f"Error: {str(e)}", "", None


def generate_kpi_cards_html(kpis):
    """Generate HTML for KPI cards"""
    cards_html = """
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
    """
    
    for key, kpi in kpis.items():
        cards_html += f"""
        <div style="
            background: linear-gradient(135deg, {kpi['color']}15, {kpi['color']}05);
            border-left: 4px solid {kpi['color']};
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; margin-bottom: 5px;">{kpi['icon']}</div>
            <div style="
                font-size: 28px;
                font-weight: bold;
                color: {kpi['color']};
                margin-bottom: 5px;
            ">{kpi['value']}</div>
            <div style="
                font-size: 12px;
                font-weight: 600;
                color: #2c3e50;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            ">{kpi['label']}</div>
            <div style="
                font-size: 11px;
                color: #7f8c8d;
            ">{kpi['subtitle']}</div>
        </div>
        """
    
    cards_html += "</div>"
    return cards_html


def create_business_charts(metrics):
    """Create comprehensive business visualization charts with tooltips"""
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('white')
    
    # Chart 1: Side-by-Side Cost Comparison Bar Chart
    ax1 = plt.subplot(131)
    
    # Data
    categories = ['Manual\nInspection', 'AI-Powered\nInspection']
    costs = [metrics['manual_annual_cost'], metrics['ai_annual_cost']]
    colors = ['#e74c3c', '#27ae60']
    
    # Create bars
    bars = ax1.bar(categories, costs, color=colors, alpha=0.8, width=0.6, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        label = f'${cost/1000:.0f}K' if cost >= 1000 else f'${cost:.0f}'
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add savings annotation
    savings = metrics['annual_savings']
    savings_pct = metrics['cost_reduction_pct']
    y_max = max(costs) * 1.2
    
    # Draw savings arrow and text
    ax1.annotate('', xy=(1, costs[1] + y_max * 0.05), xytext=(0, costs[0] - y_max * 0.05),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax1.text(0.5, (costs[0] + costs[1]) / 2 + y_max * 0.1,
            f'Save\n${savings/1000:.0f}K\n({savings_pct:.0f}%)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.2))
    
    # Styling
    ax1.set_ylabel('Annual Cost (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Annual Cost Comparison\nApple-to-Apple', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, y_max)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add methodology note
    methodology_text = f"""Method: Manual=${calculator.manual_cost_per_panel}/panel × {calculator.manual_inspections_per_year}/year
AI=${calculator.ai_cost_per_panel}/panel × {calculator.ai_inspections_per_year}/year
Savings=${savings/1000:.0f}K ({savings_pct:.0f}% reduction)"""
    ax1.text(0.5, -0.15, methodology_text, transform=ax1.transAxes,
            ha='center', va='top', fontsize=8, style='italic', color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.5))
    
    # Chart 2: Benefits Breakdown (Horizontal Bar)
    ax2 = plt.subplot(132)
    benefits = ['Cost\nSavings', 'Energy\nValue', 'Total\nBenefit']
    values = [metrics['annual_savings'], metrics['energy_value'], metrics['total_annual_benefit']]
    colors2 = ['#3498db', '#f39c12', '#1abc9c']
    
    bars = ax2.barh(benefits, values, color=colors2, alpha=0.8)
    ax2.set_xlabel('Annual Value (USD)', fontsize=11)
    ax2.set_title('Benefits Breakdown', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        label = f'${width/1000:.0f}K' if width >= 1000 else f'${width:.0f}'
        ax2.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add energy calculation note
    if 'energy_calculation_details' in metrics:
        energy_details = metrics['energy_calculation_details']
        energy_text = f"""Energy Value Calculation:
• {energy_details['panels_with_defects']:,} defective panels detected
• {energy_details['energy_saved_mwh']:.0f} MWh saved through early detection
• Based on 15% better detection, 3 months earlier
• Energy value: ${metrics['energy_value']/1000:.0f}K/year"""
        ax2.text(0.5, -0.15, energy_text, transform=ax2.transAxes,
                ha='center', va='top', fontsize=8, style='italic', color='#666',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', alpha=0.7))
    
    # Chart 3: Cost Reduction Gauge
    ax3 = plt.subplot(133)
    reduction_pct = metrics['cost_reduction_pct']
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    
    # Background arc
    ax3.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.1, color='gray')
    
    # Value arc
    value_theta = theta[int(reduction_pct)] if reduction_pct < 100 else theta[-1]
    value_arc = theta[:int(reduction_pct)+1] if reduction_pct < 100 else theta
    ax3.fill_between(np.cos(value_arc), np.sin(value_arc), 0, alpha=0.6, color='#27ae60')
    
    # Add needle
    needle_angle = np.pi * (1 - reduction_pct / 100)
    ax3.arrow(0, 0, 0.8*np.cos(needle_angle), 0.8*np.sin(needle_angle),
             head_width=0.05, head_length=0.05, fc='#2c3e50', ec='#2c3e50')
    
    # Center text
    ax3.text(0, -0.3, f'{reduction_pct:.1f}%', ha='center', va='center',
            fontsize=24, fontweight='bold', color='#27ae60')
    ax3.text(0, -0.5, 'Cost Reduction', ha='center', va='center',
            fontsize=11, color='#7f8c8d')
    
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-0.8, 1.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Cost Efficiency\nImprovement', fontsize=14, fontweight='bold', pad=20)
    
    # Add gauge interpretation note
    gauge_text = f"""Formula: (Manual Cost - AI Cost) / Manual Cost × 100
Your Result: {reduction_pct:.1f}% operational cost reduction
Interpretation: Save ${reduction_pct:.0f} of every $100 spent on manual inspection"""
    ax3.text(0.5, -0.15, gauge_text, transform=ax3.transAxes,
            ha='center', va='top', fontsize=8, style='italic', color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for notes
    return fig


def get_timer_status():
    """Get auto-shutdown timer status"""
    remaining = timer.get_remaining_time()
    minutes = int(remaining)
    seconds = int((remaining - minutes) * 60)
    
    if remaining <= 5:
        color = "#e74c3c"  # Red
    elif remaining <= 10:
        color = "#f39c12"  # Orange
    else:
        color = "#27ae60"  # Green
    
    return f"⏱️ Auto-close in: {minutes:02d}:{seconds:02d}", color


def shutdown_demo():
    """Graceful shutdown"""
    print("[INFO] Auto-shutdown triggered")
    timer.stop()
    return "Demo has been closed due to inactivity. Please refresh to restart."


# Create Gradio interface
def create_interface():
    """Create and configure Gradio interface"""
    
    # Start auto-shutdown timer
    def show_warning(msg):
        print(f"[WARNING] {msg}")
    
    timer.start(warning_callback=show_warning, shutdown_callback=shutdown_demo)
    
    with gr.Blocks(title=APP_TITLE) as demo:
        
        # Header
        gr.Markdown(APP_DESCRIPTION)
        
        # Timer status
        timer_status = gr.Markdown("⏱️ Auto-close in: 30:00", elem_id="timer")
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: Single Image
            with gr.TabItem("📷 Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Model selector
                        model_choice = gr.Dropdown(
                            choices=["SVM (96.8% accuracy)", "CNN with Grad-CAM (95.8%)"],
                            value="SVM (96.8% accuracy)",
                            label="Select Model",
                            info="SVM: Higher accuracy | CNN: Explainable with Grad-CAM"
                        )
                        model_info = gr.Markdown(update_model_info("SVM (96.8% accuracy)"))
                        
                        input_image = gr.Image(label="Upload Solar Panel Image", type="filepath")
                        predict_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            output_image = gr.Image(label="Input Image")
                            result_display = gr.HTML(label="Prediction Result")
                        
                        confidence_plot = gr.Plot(label="Confidence Scores")
                        filename_text = gr.Textbox(label="Filename", visible=False)
                
                # Grad-CAM visualization section (conditionally shown)
                with gr.Row(visible=GRADCAM_AVAILABLE) as gradcam_row:
                    gr.Markdown("### 🔍 Grad-CAM Visualization (CNN Mode Only)")
                
                with gr.Row(visible=GRADCAM_AVAILABLE):
                    gradcam_original = gr.Image(label="Original", visible=True)
                    gradcam_heatmap = gr.Image(label="Grad-CAM Heatmap", visible=True)
                    gradcam_overlay = gr.Image(label="Overlay", visible=True)
                
                # Event handlers
                model_choice.change(
                    fn=update_model_info,
                    inputs=model_choice,
                    outputs=model_info
                )
                
                predict_btn.click(
                    fn=predict_single,
                    inputs=[input_image, model_choice],
                    outputs=[output_image, result_display, confidence_plot, filename_text,
                            gradcam_original, gradcam_heatmap, gradcam_overlay]
                )
            
            # Tab 2: Batch Processing
            with gr.TabItem("📁 Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        # Model selector for batch
                        batch_model_choice = gr.Dropdown(
                            choices=["SVM (96.8% accuracy)", "CNN with Grad-CAM (95.8%)"],
                            value="SVM (96.8% accuracy)",
                            label="Select Model for Batch Processing",
                            info="Grad-CAM not available in batch mode"
                        )
                        
                        zip_input = gr.File(label="Upload ZIP file with images", file_types=['.zip'])
                        batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                    
                    with gr.Column():
                        batch_status = gr.Textbox(label="Status", value="Ready")
                        results_table = gr.DataFrame(label="Results")
                
                with gr.Row():
                    export_csv_btn = gr.Button("📊 Export CSV")
                    export_pdf_btn = gr.Button("📄 Generate PDF Report")
                    export_status = gr.Textbox(label="Export Status", value="")
                
                # File outputs for exports (need to be defined before use)
                with gr.Row():
                    csv_output = gr.File(label="Download CSV", visible=False)
                    pdf_output = gr.File(label="Download PDF", visible=False)
                
                # Event handlers
                batch_btn.click(
                    fn=predict_batch_zip,
                    inputs=[zip_input, batch_model_choice],
                    outputs=[results_table, batch_status, export_status]
                )
                
                export_csv_btn.click(fn=export_csv, outputs=[csv_output, export_status])
                export_pdf_btn.click(fn=export_pdf, outputs=[pdf_output, export_status])
            
            # Tab 3: Business Impact
            with gr.TabItem("💰 Business Impact"):
                # Header with slider
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Solar Farm ROI Calculator")
                        gr.Markdown("Calculate the business value of implementing SolarVision AI for automated defect detection.")
                    with gr.Column(scale=1):
                        farm_size_slider = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="Solar Farm Size (MW)",
                            info="Adjust to see impact on different farm sizes"
                        )
                        calc_btn = gr.Button("🚀 Calculate ROI", variant="primary", size="lg")
                
                # KPI Cards Section
                kpi_cards = gr.HTML(label="Key Metrics", value="<div style='padding: 20px; color: #7f8c8d; text-align: center;'>Click 'Calculate ROI' to see business impact metrics</div>")
                
                # Summary Section
                with gr.Row():
                    with gr.Column():
                        summary_text = gr.Textbox(
                            label="Executive Summary",
                            lines=8,
                            value="Adjust the farm size and click Calculate to see your personalized business case.",
                            interactive=False
                        )
                
                # Charts Section
                business_chart = gr.Plot(label="Business Analytics")
                
                # Key Insights
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        #### 💡 Key Benefits
                        - **Cost Reduction**: Up to 87% savings on inspection costs
                        - **Time Efficiency**: 10x faster inspection speed
                        - **Frequency**: 3x more inspections per year (monthly vs quarterly)
                        - **Accuracy**: 15% better defect detection
                        - **Safety**: Proactive identification of electrical hazards
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        #### 📈 Business Value
                        - **Immediate**: Reduced operational costs from day one
                        - **Short-term**: Payback period typically under 6 months
                        - **Long-term**: Maximize energy yield and panel lifespan
                        - **Risk Mitigation**: Prevent costly failures and downtime
                        """)
                
                calc_btn.click(
                    fn=calculate_business,
                    inputs=farm_size_slider,
                    outputs=[kpi_cards, summary_text, business_chart]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        **SolarVision AI** | Automated PV Panel Defect Detection System  
        Dual Model Support: SVM (96.8%) & CNN with Grad-CAM (95.8%) | Dataset: Alicja Lenarczyk, PhD
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch app
    app = create_interface()
    
    # Launch with Hugging Face compatible settings
    import tempfile
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Generate public link
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(font=gr.themes.GoogleFont("Inter")),
        allowed_paths=[tempfile.gettempdir(), "outputs"]  # Allow access to temp and outputs directories
    )
