import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle


def create_pdf(prediction, confidence, image_type, original_path, heatmap_path, nutrition_text = None, report_name = 'AnemoScan_Report', full_original_path = None):
    
    pdf_path = f'{report_name}.pdf'
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize = A4,
        topMargin = 0.5 * inch,
        bottomMargin = 0.5 * inch,
        leftMargin = 0.7 * inch,
        rightMargin = 0.7 * inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent = styles['Title'],
        fontSize = 18,
        textColor = colors.HexColor('#2c3e50'),
        alignment = TA_CENTER,
        spaceAfter = 12
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent = styles['Heading2'],
        fontSize = 14,
        textColor = colors.HexColor('#c0392b'),
        spaceBefore = 12,
        spaceAfter = 6
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent = styles['Normal'],
        fontSize = 10,
        leading = 14,
        alignment = TA_LEFT
    )
    
    # For table cells, we need a style that handles bold properly
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent = normal_style,
        fontSize = 10,
        leading = 12
    )
    
    content = []
    
    # Logo (optional)
    logo_path = 'logo.png'
    if os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width = 150, height = 150, hAlign = 'LEFT')
            content.append(logo)
            content.append(Spacer(1, 0.2 * inch))
        except:
            pass
    
    # Title and date
    content.append(Paragraph('AnemoScan Medical Report', title_style))
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph(f'Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}', normal_style))
    content.append(Spacer(1, 0.2 * inch))
    
    # Results Summary Table
    
    # Prepare cell values as Paragraphs
    param_header = Paragraph('<b>Parameter</b>', table_cell_style)
    value_header = Paragraph('<b>Value</b>', table_cell_style)
    
    # Prediction cell with color
    if prediction == 'Anemic':
        pred_text = f'<font color="#e74c3c"><b>{prediction}</b></font>'
    else:
        pred_text = f'<font color="#27ae60"><b>{prediction}</b></font>'
    pred_cell = Paragraph(pred_text, table_cell_style)
    
    confidence_cell = Paragraph(f'{confidence*100:.1f}%', table_cell_style)
    image_type_cell = Paragraph(image_type, table_cell_style)
    
    # Build table data
    data = [
        [param_header, value_header],
        [Paragraph('Prediction', table_cell_style), pred_cell],
        [Paragraph('Confidence', table_cell_style), confidence_cell],
        [Paragraph('Image Type', table_cell_style), image_type_cell],
    ]
    
    # Create table with wider columns
    table = Table(data, colWidths = [2.0 * inch, 3.0 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#f0f0f0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(table)
    content.append(Spacer(1, 0.2 * inch))
    
    # Interpretation
    content.append(Paragraph('Interpretation', heading_style))
    if prediction == 'Anemic':
        interpret_text = 'The AI model detected visual patterns consistent with anemia (pallor). The heatmap highlights the regions (conjunctiva/nail) that most influenced this decision.'
    else:
        interpret_text = 'The AI model did not detect significant pallor. The heatmap shows the model focused on normal coloration patterns.'
    content.append(Paragraph(interpret_text, normal_style))
    content.append(Spacer(1, 0.15 * inch))
    
    # Nutrition guidance
    content.append(Paragraph('Nutrition Recommendations', heading_style))
    if nutrition_text is None:
        if prediction == 'Anemic':
            nutrition_html = """
            <b>Iron-rich foods:</b><br/>
            • Spinach, kale, broccoli<br/>
            • Red meat, liver (in moderation)<br/>
            • Lentils, chickpeas, beans<br/>
            <b>Vitamin C (enhances absorption):</b><br/>
            • Oranges, lemons, grapefruit<br/>
            • Strawberries, kiwi<br/>
            • Bell peppers<br/>
            <b>Avoid with iron meals:</b><br/>
            • Tea and coffee (reduce absorption)<br/>
            <b>Tip:</b> Combine iron + vitamin C (e.g., lentils + lemon juice)
            """
        else:
            nutrition_html = 'Continue a balanced diet rich in iron and vitamins to maintain healthy hemoglobin levels.'
    else:
        # Convert plain text newlines to <br/> and handle basic formatting
        nutrition_html = nutrition_text.replace('\n', '<br/>')
        # Replace markdown style *bold* with <b>
        import re
        nutrition_html = re.sub(r'\\(.?)\\*', r'<b>\1</b>', nutrition_html)
    
    content.append(Paragraph(nutrition_html, normal_style))
    content.append(Spacer(1, 0.15 * inch))
    
    # Images
    content.append(Paragraph('Analysis Visuals', heading_style))

    # Full uploaded image
    if full_original_path and os.path.exists(full_original_path):
        content.append(Paragraph('<b>Full Uploaded Image</b>', normal_style))
        content.append(RLImage(full_original_path, width = 3.5 * inch, height = 3.5 * inch))
        content.append(Spacer(1, 0.1 * inch))

    # Cropped ROI (used by the model)
    content.append(Paragraph('<b>Cropped Region of Interest (ROI)</b>', normal_style))
    content.append(RLImage(original_path, width = 2.5 * inch, height = 2.5 * inch))
    content.append(Spacer(1, 0.1 * inch))

    # Heatmap
    content.append(Paragraph('<b>Grad-CAM Heatmap (Overlay on ROI)</b>', normal_style))
    content.append(RLImage(heatmap_path, width = 2.5 * inch, height = 2.5 * inch))
    content.append(Spacer(1, 0.2 * inch))
    
    # Disclaimer
    content.append(Paragraph('Disclaimer', heading_style))
    disclaimer_text = """
    This system is not a medical diagnostic tool and is intended for educational and screening purposes only. 
    Always consult a qualified healthcare professional for any medical concerns or before making health decisions.
    """
    content.append(Paragraph(disclaimer_text, normal_style))
    
    doc.build(content)
    return pdf_path