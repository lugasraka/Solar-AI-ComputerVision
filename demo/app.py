"""
SolarVision AI - Gradio Demo Application
Interactive web interface for PV panel defect detection
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
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from inference import get_predictor
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

**Model**: ResNet18 + SVM (96.8% accuracy on test set)

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


def predict_single(image):
    """Process single image"""
    global processed_results
    
    # Reset timer on activity
    timer.reset()
    
    if image is None:
        return None, "Please upload an image", "", ""
    
    try:
        # Handle different image input types from Gradio
        if isinstance(image, str):
            # Image is already a file path
            temp_path = Path(image)
        elif isinstance(image, np.ndarray):
            # Image is a numpy array, save it
            temp_path = Path(tempfile.gettempdir()) / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            Image.fromarray(image).save(temp_path)
        elif hasattr(image, 'save'):
            # Image is a PIL Image object
            temp_path = Path(tempfile.gettempdir()) / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image.save(temp_path)
        else:
            return None, f"Unsupported image type: {type(image)}", "", ""
        
        # Get prediction
        pred = load_predictor().predict(str(temp_path))
        processed_results = [pred]
        
        # Create results display
        result_html = f"""
        <div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa;'>
            <h3 style='color: {get_confidence_color(pred["confidence"])};'>
                Predicted: {pred['predicted_class']}
            </h3>
            <p style='font-size: 18px;'>
                Confidence: <strong>{format_confidence(pred['confidence'])}</strong>
            </p>
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
        
        # Load image for display (if it was a temp file we created, keep it for display then clean up)
        if isinstance(image, str):
            # Original was a file path, load it for display
            display_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            display_image = image
        else:
            display_image = image
        
        # Clean up temp file only if we created it
        if not isinstance(image, str):
            temp_path.unlink(missing_ok=True)
        
        return display_image, result_html, fig, pred['filename']
        
    except Exception as e:
        return image, f"Error: {str(e)}", None, ""


def predict_batch_zip(zip_file, progress=gr.Progress()):
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
        
        # Process images
        results = []
        tracker = ProgressTracker(len(image_files))
        
        for i, img_path in enumerate(image_files):
            try:
                pred = load_predictor().predict(str(img_path))
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
                    'Top Class': r['top3'][0][0] if r['top3'] else 'N/A'
                })
        
        df = pd.DataFrame(df_data)
        
        # Summary text
        summary = f"Processed {len(results)} images successfully"
        if any('error' in r for r in results):
            errors = sum(1 for r in results if 'error' in r)
            summary += f" ({errors} errors)"
        
        return df, summary, "Ready to export"
        
    except Exception as e:
        return pd.DataFrame(), f"Error processing ZIP: {str(e)}", "Error"


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
    """Calculate business impact"""
    timer.reset()
    
    try:
        metrics = calculator.calculate(farm_size)
        report_text = calculator.generate_report_text(metrics)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cost comparison
        cost_labels = ['Manual\nInspection', 'AI-Powered\nInspection']
        cost_values = [metrics['manual_annual_cost'], metrics['ai_annual_cost']]
        bars1 = ax1.bar(cost_labels, cost_values, color=['#e74c3c', '#27ae60'])
        ax1.set_ylabel('Annual Cost (USD)', fontsize=12)
        ax1.set_title('Cost Comparison', fontsize=14, fontweight='bold')
        ax1.set_yticklabels([f'${x/1000:.0f}K' for x in ax1.get_yticks()])
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}K',
                    ha='center', va='bottom', fontsize=10)
        
        # Benefits breakdown
        benefit_labels = ['Cost\nSavings', 'Energy\nValue']
        benefit_values = [metrics['annual_savings'], metrics['energy_value']]
        bars2 = ax2.bar(benefit_labels, benefit_values, color=['#3498db', '#f39c12'])
        ax2.set_ylabel('Annual Value (USD)', fontsize=12)
        ax2.set_title('Annual Benefits', fontsize=14, fontweight='bold')
        ax2.set_yticklabels([f'${x/1000:.0f}K' for x in ax2.get_yticks()])
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}K',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        return report_text, fig, metrics['annual_savings'], metrics['time_saved_hours']
        
    except Exception as e:
        return f"Error: {str(e)}", None, 0, 0


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
    # Note: Gradio doesn't support programmatic shutdown, 
    # but we can show a message
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
                        input_image = gr.Image(label="Upload Solar Panel Image", type="filepath")
                        predict_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Input Image")
                        result_display = gr.HTML(label="Prediction Result")
                        confidence_plot = gr.Plot(label="Confidence Scores")
                        filename_text = gr.Textbox(label="Filename", visible=False)
                
                predict_btn.click(
                    fn=predict_single,
                    inputs=input_image,
                    outputs=[output_image, result_display, confidence_plot, filename_text]
                )
            
            # Tab 2: Batch Processing
            with gr.TabItem("📁 Batch Processing"):
                with gr.Row():
                    with gr.Column():
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
                    inputs=zip_input,
                    outputs=[results_table, batch_status, export_status]
                )
                
                export_csv_btn.click(fn=export_csv, outputs=[csv_output, export_status])
                export_pdf_btn.click(fn=export_pdf, outputs=[pdf_output, export_status])
            
            # Tab 3: Business Impact
            with gr.TabItem("💰 Business Impact"):
                with gr.Row():
                    with gr.Column():
                        farm_size_slider = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="Solar Farm Size (MW)"
                        )
                        calc_btn = gr.Button("📊 Calculate ROI", variant="primary")
                    
                    with gr.Column():
                        savings_display = gr.Number(label="Annual Savings (USD)", value=0)
                        time_saved_display = gr.Number(label="Hours Saved per Year", value=0)
                
                business_report = gr.Textbox(label="Business Report", lines=20)
                business_chart = gr.Plot(label="Cost Analysis")
                
                calc_btn.click(
                    fn=calculate_business,
                    inputs=farm_size_slider,
                    outputs=[business_report, business_chart, savings_display, time_saved_display]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        **SolarVision AI** | Automated PV Panel Defect Detection System  
        Model: ResNet18 + SVM (96.8% accuracy) | Dataset: Alicja Lenarczyk, PhD
        """)
        
        # Timer will update on user interactions only
        # Note: Auto-shutdown works in background but display updates on activity
    
    return demo


if __name__ == "__main__":
    # Create and launch app
    app = create_interface()
    
    # Launch with Hugging Face compatible settings
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Generate public link
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft()
    )
