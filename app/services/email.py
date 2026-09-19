import os
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAILS_FROM_NAME = os.getenv("EMAILS_FROM_NAME", "MzaziCare Health")

def send_verification_email(to_email: str, full_name: str, otp_code: str):
    """Sends an HTML email with the 6-digit OTP code"""

    # Fallback if credentials aren't configured yet
    if not SMTP_USER or not SMTP_PASSWORD:
        print(f"[MAIL MOCK] SMTP not configured. OTP for {to_email} is: {otp_code}", flush=True)
        return

    msg = EmailMessage()
    msg["Subject"] = f"{otp_code} is your MzaziCare verification code"
    msg["From"] = f"{EMAILS_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_email

    # Plaintext fallback
    msg.set_content(
        f"Hello {full_name}, \n\n"
        f"Your MzaziCare verification code is: {otp_code}\n\n"
        f"This code will expire in 10 minutes.\n"
        f"If you did not request this, please ignore this email."
    )

    # Professional HTML layout
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f3f4f6; margin:0; padding: 30px;">
            <div style="max-width: 500px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; padding: 32px; border: 1px solid #e5e7eb;">
                <h2 style="color: #111827; margin-top: 0;">Verify Your Email</h2>
                <p style="color: #4b5563; font-size: 15px; line-height: 24px;">
                    Hello <strong>{full_name}</strong>,<br/>
                    Thank you for registereing on the MzaziCare portal. Use the 6-digit code below to confirm your email and complete your enrollment:
                </p>

                <div style="background-color: #eef2ff; border: 1px dashed #6366f1; border-radius: 8px; padding: 16px; text-align: center; margin: 24px 0;">
                    <span style="font-size: 32px; font-weight: 700; letter-spacing: 8px; color: #4338ca;">
                        {otp_code}
                    </span>
                </div>

                <p style="color: #6b7280; font-size: 13px; margin-bottom: 0;">
                    This verification code is valid for <strong>10 minutes</strong>/ If you did not request this registration, no action is needed.
                </p>
            </div>
        </body>
    </html>
    """
    msg.add_alternative(html_content, subtype="html")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
            print(f" Email sent to {to_email}", flush=True)
    except Exception as e:
        print(f" Failed to send email to {to_email}: {e}", flush=True)