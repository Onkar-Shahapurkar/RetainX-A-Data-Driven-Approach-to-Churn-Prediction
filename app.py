from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import datetime
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model = pickle.load(open('Model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        
        # Convert string values to numeric using LabelEncoder
        encoder = LabelEncoder()
        for feature in df.columns:
            if df[feature].dtype == 'object':
                df[feature] = encoder.fit_transform(df[feature])
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1][0]
        
        # Format response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'churn_risk': 'High' if prediction[0] == 1 else 'Low',
            'confidence': round(float(probability) * 100, 2) if prediction[0] == 1 else round((1 - float(probability)) * 100, 2)
        }
        
        # Determine top risk factors
        risk_factors = []
        
        if data.get('contract') == 'Month-to-month':
            risk_factors.append({
                'factor': 'Month-to-month contract',
                'impact': 'High',
                'recommendation': 'Offer incentives for longer contract terms'
            })
            
        if data.get('onlineSecurity') == 'No':
            risk_factors.append({
                'factor': 'No online security service',
                'impact': 'Medium',
                'recommendation': 'Provide free trial of security services'
            })
            
        if data.get('techSupport') == 'No':
            risk_factors.append({
                'factor': 'No tech support',
                'impact': 'Medium',
                'recommendation': 'Offer complimentary tech support sessions'
            })
            
        if int(data.get('tenure', 0)) < 12:
            risk_factors.append({
                'factor': 'New customer (< 1 year)',
                'impact': 'High',
                'recommendation': 'Implement new customer loyalty program'
            })
            
        if float(data.get('monthlyCharges', 0)) > 70:
            risk_factors.append({
                'factor': 'High monthly charges',
                'impact': 'Medium',
                'recommendation': 'Review pricing plan and offer temporary discounts'
            })
        
        response['risk_factors'] = risk_factors
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analytics')
def analytics():
    # In a real application, this would pull from your database
    # For demo purposes, we're returning static data
    churn_data = {
        'total_customers': 7043,
        'churn_rate': 26.5,
        'monthly_charges_avg': 64.76,
        'contract_distribution': {
            'month_to_month': 55,
            'one_year': 21,
            'two_year': 24
        },
        'churn_by_contract': {
            'month_to_month': 42.7,
            'one_year': 11.3,
            'two_year': 2.8
        },
        'churn_by_tenure': {
            '0-6': 41.7,
            '7-12': 35.2,
            '13-18': 28.4,
            '19-24': 22.1,
            '25-36': 15.7,
            '37-48': 10.8,
            '49-60': 7.5,
            '61+': 5.2
        }
    }
    
    return jsonify(churn_data)

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    # Get data from request
    data = request.json
    customer_data = data.get('customer_data', {})
    prediction_result = data.get('prediction_result', {})
    
    # Create an in-memory buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF with ReportLab
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    # Create a list to hold the elements that will go in the PDF
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MyTitle',
                              fontName='Helvetica-Bold',
                              fontSize=18,
                              alignment=TA_CENTER,
                              spaceAfter=12))
    styles.add(ParagraphStyle(name='Subtitle',
                              fontName='Helvetica-Bold',
                              fontSize=14,
                              spaceBefore=12,
                              spaceAfter=6))
    styles.add(ParagraphStyle(name='MyNormal',
                              fontName='Helvetica',
                              fontSize=10,
                              spaceBefore=6,
                              spaceAfter=6))
    styles.add(ParagraphStyle(name='Alert',
                              fontName='Helvetica-Bold',
                              fontSize=12,
                              textColor=colors.red,
                              spaceBefore=6,
                              spaceAfter=6))
    styles.add(ParagraphStyle(name='Success',
                              fontName='Helvetica-Bold',
                              fontSize=12,
                              textColor=colors.green,
                              spaceBefore=6,
                              spaceAfter=6))
    
    # Add title
    elements.append(Paragraph('Customer Churn Prediction Report', styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Add date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {current_datetime}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Add summary
    is_high_risk = "High" in prediction_result.get('prediction', '')
    if is_high_risk:
        risk_text = "This customer has been identified as having a <b>HIGH RISK</b> of churning."
        risk_style = styles['Alert']
    else:
        risk_text = "This customer has been identified as having a <b>LOW RISK</b> of churning."
        risk_style = styles['Success']
    
    elements.append(Paragraph("Executive Summary", styles['Subtitle']))
    elements.append(Paragraph(risk_text, risk_style))
    elements.append(Paragraph(f"Confidence level: {prediction_result.get('confidence', 0)}%", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Add customer data section
    elements.append(Paragraph("Customer Profile", styles['Subtitle']))
    
    # Create formatted customer data table
    customer_data_table = [
        ["Parameter", "Value"],
        ["Contract Type", customer_data.get('contract', 'N/A')],
        ["Tenure (Months)", customer_data.get('tenure', 'N/A')],
        ["Monthly Charges ($)", customer_data.get('monthlyCharges', 'N/A')],
        ["Total Charges ($)", customer_data.get('totalCharges', 'N/A')],
        ["Dependents", customer_data.get('dependents', 'N/A')],
        ["Online Security", customer_data.get('onlineSecurity', 'N/A')],
        ["Online Backup", customer_data.get('onlineBackup', 'N/A')],
        ["Device Protection", customer_data.get('deviceProtection', 'N/A')],
        ["Tech Support", customer_data.get('techSupport', 'N/A')],
        ["Paperless Billing", customer_data.get('paperlessBilling', 'N/A')]
    ]
    
    t = Table(customer_data_table, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Add Risk Factors section
    elements.append(Paragraph("Risk Factors Analysis", styles['Subtitle']))
    
    if 'risk_factors' in prediction_result and isinstance(prediction_result['risk_factors'], list):
        risk_factors = prediction_result['risk_factors']
        
        if risk_factors:
            # Create risk factors table data
            risk_data = [["Risk Factor", "Impact", "Recommended Action"]]
            
            # Map text descriptions to impact values
            impact_map = {
                "Month-to-month contract": "High",
                "No online security service": "Medium",
                "No tech support": "Medium",
                "New customer (< 1 year)": "High",
                "High monthly charges": "Medium"
            }
            
            # Map impact to recommendations
            recommendations = {
                "Month-to-month contract": "Offer discounts for 1 or 2-year contract commitments. Consider customized pricing plans with gradual increases.",
                "No online security service": "Provide a free 3-month trial of online security services. Highlight the value and protection benefits.",
                "No tech support": "Offer complimentary tech support sessions. Create an easy-to-access support knowledge base.",
                "New customer (< 1 year)": "Implement a first-year loyalty program with exclusive benefits. Schedule regular check-in calls.",
                "High monthly charges": "Review pricing plan. Consider bundled services at a discount or temporary promotional offers."
            }
            
            for factor in risk_factors:
                factor_text = factor  # If factor is a string
                impact = "Medium"     # Default impact
                recommendation = "Monitor and assess"  # Default recommendation
                
                # Extract data from the factor based on its format
                for factor_key, impact_value in impact_map.items():
                    if factor_key in factor:
                        factor_text = factor_key
                        impact = impact_value
                        recommendation = recommendations.get(factor_key, recommendation)
                        break
                
                risk_data.append([factor_text, impact, recommendation])
            
            # Create the risk factors table
            risk_table = Table(risk_data, colWidths=[2*inch, 1*inch, 2.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Color code the impact column
                ('TEXTCOLOR', (1, 1), (1, -1), colors.black),
            ]))
            
            # Apply conditional formatting for impact column
            for i in range(1, len(risk_data)):
                impact = risk_data[i][1]
                if impact == 'High':
                    risk_table.setStyle(TableStyle([
                        ('BACKGROUND', (1, i), (1, i), colors.lightcoral)
                    ]))
                elif impact == 'Medium':
                    risk_table.setStyle(TableStyle([
                        ('BACKGROUND', (1, i), (1, i), colors.lightyellow)
                    ]))
                else:
                    risk_table.setStyle(TableStyle([
                        ('BACKGROUND', (1, i), (1, i), colors.lightgreen)
                    ]))
            
            elements.append(risk_table)
        else:
            elements.append(Paragraph("No significant risk factors identified.", styles['Normal']))
    else:
        elements.append(Paragraph("Risk factor data not available.", styles['Normal']))
    
    elements.append(Spacer(1, 24))
    
    # Add recommended action plan section
    elements.append(Paragraph("Recommended Retention Strategy", styles['Subtitle']))
    
    if is_high_risk:
        elements.append(Paragraph("<b>Priority Level: High</b> - Immediate action recommended", styles['Alert']))
        elements.append(Spacer(1, 6))
        
        # Create action plan based on risk factors
        action_items = [
            "Schedule a customer service call within the next 7 days",
            "Review customer usage patterns and offer tailored service packages",
            "Provide a personalized retention offer based on the identified risk factors"
        ]
        
        for i, item in enumerate(action_items):
            elements.append(Paragraph(f"{i+1}. {item}", styles['Normal']))
        
        # Add personalized recommendation based on contract type
        if customer_data.get('contract') == 'Month-to-month':
            elements.append(Paragraph("<b>Special Recommendation:</b> Offer a 15% discount for the first 6 months with a 1-year contract commitment.", styles['Normal']))
    else:
        elements.append(Paragraph("<b>Priority Level: Low</b> - Regular monitoring recommended", styles['Success']))
        elements.append(Spacer(1, 6))
        
        # Recommendations for low-risk customers
        elements.append(Paragraph("Recommendations:", styles['Normal']))
        elements.append(Paragraph("1. Include customer in regular satisfaction surveys", styles['Normal']))
        elements.append(Paragraph("2. Offer loyalty rewards for continued service", styles['Normal']))
        elements.append(Paragraph("3. Consider cross-selling additional services at next billing cycle", styles['Normal']))
    
    elements.append(Spacer(1, 24))
    
    # Add footer with disclaimer
    elements.append(Paragraph("<i>Disclaimer: This prediction is based on machine learning models and historical data patterns. The actual customer behavior may vary. This report should be used as a decision support tool, not as the sole basis for business decisions.</i>", styles['Normal']))
    
    # Build the PDF document
    doc.build(elements)
    
    # Move to the beginning of the buffer
    buffer.seek(0)
    
    # Return the PDF as a downloadable file
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='ChurnPrediction_Report.pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)