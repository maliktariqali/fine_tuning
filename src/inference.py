"""
Simple inference script for the IT Helpdesk Assistant.

Run after training:
    python src/inference.py

If the DPO adapter is not available locally, the script falls back to a
small retrieval baseline over the assignment policy answers so the demo
still works while you are preparing the GPU run.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL_DIR = ROOT / "unsloth_it_support_merge_reload_outputs" / "stage3_dpo_final_merged_model"
BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
SYSTEM_PROMPT = (
    "You are HelpDeskAI, an internal IT helpdesk assistant. "
    "Answer clearly, follow company policy, protect credentials and confidential data, "
    "and ask for ticket details when escalation is needed."
)

POLICIES = [
    {
        "question": "How do I reset my company password?",
        "answer": "Open the self-service password portal, verify your identity with MFA, and choose a new password that meets the company policy. If the portal fails or your MFA device is unavailable, contact the IT helpdesk so an agent can verify your identity and issue a temporary reset link.",
        "title": "Password reset"
    },
    {
        "question": "My account is locked after too many login attempts. What should I do?",
        "answer": "Wait 15 minutes for the automatic lockout window to expire, then sign in with the correct password. If the lockout continues, stop retrying and contact IT because a stale password on a phone, mail app, or VPN client may be repeatedly locking the account.",
        "title": "Account lockout"
    },
    {
        "question": "How can I enroll a new phone for multi-factor authentication?",
        "answer": "Go to the security portal, choose MFA devices, and add the new phone using the authenticator app or SMS option approved by IT. Keep your old MFA method active until the new device is verified, then remove the old device if it is no longer used.",
        "title": "MFA enrollment"
    },
    {
        "question": "I lost my MFA phone and cannot sign in. What is the correct process?",
        "answer": "Call the IT helpdesk from a registered number or join the verified identity process with your manager. IT will revoke the lost device, issue a temporary sign-in method, and require you to enroll a new MFA device immediately.",
        "title": "MFA lost device"
    },
    {
        "question": "How do I connect to the company VPN?",
        "answer": "Install the approved VPN client, sign in with your company email, approve MFA, and select the nearest corporate gateway. If you cannot connect, check your internet connection first and then raise a ticket with the VPN error code and timestamp.",
        "title": "VPN access"
    },
    {
        "question": "The VPN client says authentication failed. How should I troubleshoot it?",
        "answer": "Confirm that your password works in the web portal and that MFA approval is being sent to the correct device. If authentication still fails, collect the VPN error message, time of failure, and network type, then send those details to IT.",
        "title": "VPN failure"
    },
    {
        "question": "How do I connect to office Wi-Fi?",
        "answer": "Choose the corporate Wi-Fi network, sign in with your company credentials, and approve any certificate prompt only if it shows the company certificate authority. Visitors should use the guest Wi-Fi network instead of the corporate network.",
        "title": "Wi-Fi access"
    },
    {
        "question": "Can I give a visitor access to Wi-Fi?",
        "answer": "Yes, but visitors must use the guest Wi-Fi network. Create or request a guest access code through the visitor portal, set the expiry date, and do not share corporate Wi-Fi credentials with any visitor.",
        "title": "Guest Wi-Fi"
    },
    {
        "question": "How do I request a new software installation?",
        "answer": "Submit a software request ticket with the business reason, software name, version, license requirement, and manager approval if the tool is paid. IT will check security approval and licensing before installing it.",
        "title": "Software installation"
    },
    {
        "question": "Where can I check whether a software tool is approved?",
        "answer": "Check the approved software catalog in the IT service portal. If the tool is not listed, submit a review request with the vendor name, purpose, data handled, and license details before installing it.",
        "title": "Approved software list"
    },
    {
        "question": "Can I get local admin rights on my laptop?",
        "answer": "Local admin rights are granted only for approved business needs and usually for a limited time. Submit a privileged access request with the task, duration, and manager approval; IT may provide temporary elevation instead of permanent admin access.",
        "title": "Admin rights"
    },
    {
        "question": "My laptop shows an endpoint security alert. What should I do?",
        "answer": "Do not ignore or close the alert without reading it. Disconnect from public Wi-Fi if you suspect malware, take a screenshot of the alert, and contact IT immediately so security can review the event.",
        "title": "Endpoint security alert"
    },
    {
        "question": "What should I do if I receive a suspicious email?",
        "answer": "Do not click links, open attachments, or reply. Use the report phishing button in the mail client, then delete the message after it is reported. If you already clicked a link or entered credentials, contact IT security immediately.",
        "title": "Phishing email"
    },
    {
        "question": "Why was my email attachment blocked?",
        "answer": "Attachments may be blocked if they contain executable files, macros, encrypted archives, or malware indicators. Ask the sender to use the approved secure transfer tool or submit the attachment for security review if it is business critical.",
        "title": "Email attachment blocked"
    },
    {
        "question": "How do I get access to a shared mailbox?",
        "answer": "Submit an access request with the shared mailbox name, required permission level, and approval from the mailbox owner. IT will add access through the identity system and it may take up to one hour to appear in the mail client.",
        "title": "Shared mailbox access"
    },
    {
        "question": "My mailbox is full. How can I fix it?",
        "answer": "Archive old messages, delete large unnecessary attachments, and empty deleted items. If the mailbox is business critical and still near the limit, request a quota review through the service portal with justification.",
        "title": "Mailbox quota"
    },
    {
        "question": "How can I create a distribution list?",
        "answer": "Open a collaboration request ticket with the list name, purpose, owner, membership type, and whether external senders are allowed. IT will create the list only after ownership and access rules are clear.",
        "title": "Distribution list"
    },
    {
        "question": "The office printer is not printing my document. What should I check?",
        "answer": "Confirm that you selected the correct printer queue, clear any paused jobs, and check whether the printer shows a paper, toner, or network error. If the issue remains, raise a ticket with the printer name, floor, and error message.",
        "title": "Printer issue"
    },
    {
        "question": "My company laptop is overheating and shutting down. What should I do?",
        "answer": "Save your work, shut down the laptop, unplug non-essential peripherals, and let it cool in a safe place. Raise a hardware ticket with the asset tag and symptoms; do not continue using a device that repeatedly overheats.",
        "title": "Laptop hardware issue"
    },
    {
        "question": "How do I request a replacement laptop or accessory?",
        "answer": "Submit a hardware request with the asset tag, issue details, business impact, and manager approval if the item is an upgrade or non-standard accessory. IT will decide whether to repair, replace, or reassign equipment.",
        "title": "Hardware replacement"
    },
    {
        "question": "My monitor does not work through the docking station. How can I troubleshoot it?",
        "answer": "Check that the dock has power, reconnect the USB-C cable, and test the monitor directly with the laptop if possible. If the issue continues, provide the dock model, monitor model, cable type, and whether other ports work.",
        "title": "Docking station"
    },
    {
        "question": "Where should I store work files so they are backed up?",
        "answer": "Store work files in the approved cloud drive or team site, not only on the local desktop. Files saved only on a laptop may be lost if the device is damaged, replaced, or wiped during a security incident.",
        "title": "Data backup"
    },
    {
        "question": "I deleted an important file from the cloud drive. Can IT recover it?",
        "answer": "Check the recycle bin or version history in the cloud drive first. If the file is not available there, open a recovery ticket with the file name, location, owner, and approximate deletion time as soon as possible.",
        "title": "File recovery"
    },
    {
        "question": "Can I share company files with an external partner?",
        "answer": "Use the approved cloud sharing workflow and choose the least-permissive access needed. Confirm that the file is allowed for external sharing, set an expiry date where possible, and avoid public anonymous links unless explicitly approved.",
        "title": "Cloud sharing"
    },
    {
        "question": "What is the safest way to send a confidential file?",
        "answer": "Use the approved secure transfer tool or encrypted cloud-sharing workflow with named recipients, access expiry, and download restrictions where available. Do not send confidential files through personal email or public file-sharing services.",
        "title": "Secure file transfer"
    },
    {
        "question": "How do I know whether a document is confidential?",
        "answer": "Use the data classification labels in the document toolbar or company policy guide. If the file contains customer data, employee records, financial details, credentials, legal content, or non-public business plans, treat it as confidential until the owner confirms otherwise.",
        "title": "Data classification"
    },
    {
        "question": "How does IT decide ticket priority?",
        "answer": "Priority is based on business impact and urgency. A single user with a workaround is usually lower priority, while an outage affecting many users, customer operations, payroll, security, or executive support is escalated faster.",
        "title": "Ticket priority"
    },
    {
        "question": "When should an IT issue be escalated?",
        "answer": "Escalate when the issue affects a critical system, many users, a security risk, a customer commitment, or an executive event. Include impact, affected users, deadlines, error messages, and any workaround already tried.",
        "title": "Incident escalation"
    },
    {
        "question": "How can I check whether a company system is down?",
        "answer": "Check the IT service status page before raising a new ticket. If there is an active incident, follow the posted workaround and updates; if your issue is different, raise a ticket and mention that the status page was checked.",
        "title": "Service status"
    },
    {
        "question": "Can I use remote desktop to access my office computer?",
        "answer": "Remote desktop is allowed only through the approved VPN or zero-trust access method and only for devices assigned to you. Do not expose remote desktop directly to the internet or share remote access credentials.",
        "title": "Remote desktop"
    },
    {
        "question": "How do I access company email on my phone?",
        "answer": "Install the approved mobile device management profile and company mail app, then sign in with MFA. The device must meet security requirements such as screen lock, encryption, and the ability for IT to remove company data if needed.",
        "title": "Mobile device enrollment"
    },
    {
        "question": "What should I do if my laptop or phone is lost?",
        "answer": "Report the loss to IT security immediately with the device type, asset tag if known, last known location, and time of loss. IT will lock accounts if needed, locate or wipe managed devices, and guide you through replacement steps.",
        "title": "Lost device"
    },
    {
        "question": "How do I request access to a business application?",
        "answer": "Submit an access request with the application name, role needed, business reason, and approval from the application owner or your manager. IT grants access through approved groups and removes it when it is no longer needed.",
        "title": "Access request"
    },
    {
        "question": "Why was I added to an access group instead of getting direct permissions?",
        "answer": "Group-based access is easier to audit, review, and remove when roles change. Direct permissions are avoided because they are harder to track and can create long-term access that no one owns.",
        "title": "Permission group"
    },
    {
        "question": "What information is needed to create a new employee account?",
        "answer": "HR or the hiring manager must provide the employee name, start date, role, department, manager, location, and required applications. IT creates accounts from the onboarding request and grants standard role-based access.",
        "title": "Onboarding account"
    },
    {
        "question": "What happens to IT access when an employee leaves?",
        "answer": "Access is disabled according to the offboarding date and urgency provided by HR. IT revokes accounts, transfers ownership where approved, collects equipment, and preserves data according to retention requirements.",
        "title": "Offboarding access"
    },
    {
        "question": "My microphone does not work in video meetings. What should I check?",
        "answer": "Check the meeting app input device, operating system privacy permissions, headset connection, and whether another app is using the microphone. If the problem continues, share the meeting app, headset model, and screenshot of audio settings with IT.",
        "title": "Video meeting issue"
    },
    {
        "question": "A web application is not loading correctly. Should I clear my browser cache?",
        "answer": "Yes, clearing cache and cookies for the affected site can fix stale sessions or outdated files. If the issue continues, test another browser and send IT the URL, screenshot, error text, and time of failure.",
        "title": "Browser cache"
    },
    {
        "question": "What should I do if single sign-on is not working?",
        "answer": "Check the service status page and avoid repeated login attempts. If no incident is posted, raise a high-impact ticket with affected applications, screenshots, error codes, and whether other users are affected.",
        "title": "SSO outage"
    },
    {
        "question": "I cannot access a network drive. What details should I send IT?",
        "answer": "Send the drive path, screenshot of the error, whether you are on VPN or office network, whether colleagues can access it, and when it last worked. This helps IT separate permission issues from network or server issues.",
        "title": "Network drive"
    },
    {
        "question": "Can I store API keys in a document or spreadsheet?",
        "answer": "No. API keys, tokens, and secrets must be stored in the approved secrets manager or secure application configuration. If a key was stored in a document or shared accidentally, rotate it and notify IT security.",
        "title": "API credentials"
    },
    {
        "question": "Should I use the company password manager?",
        "answer": "Yes, use the approved password manager for work credentials and shared team secrets. Do not reuse passwords across systems, and never store passwords in spreadsheets, notes, chat, or personal password managers.",
        "title": "Password manager"
    },
    {
        "question": "Why does IT ask me to restart my laptop after updates?",
        "answer": "Some security patches do not fully apply until the device restarts. Restart your laptop during the maintenance window or as soon as practical so the device remains protected and compliant.",
        "title": "Patch restart"
    },
    {
        "question": "Can I use a USB drive for company files?",
        "answer": "USB storage is restricted and should not be used for confidential company files unless IT has approved an encrypted device for a specific business need. Use approved cloud storage or secure transfer tools instead.",
        "title": "USB storage"
    },
    {
        "question": "Where should I look before opening a helpdesk ticket?",
        "answer": "Search the IT knowledge base and service status page first. If the article does not solve your issue, open a ticket and mention the article you tried so the helpdesk can avoid repeating the same steps.",
        "title": "Knowledge base"
    },
    {
        "question": "How do I request a change to a production system?",
        "answer": "Submit a change request with the business reason, technical plan, rollback plan, risk level, test evidence, planned window, and approver. Production changes should not be made outside the approved change process.",
        "title": "Change request"
    },
    {
        "question": "A business application is giving an error. What should my ticket include?",
        "answer": "Include the application name, exact error text, screenshot, time of issue, action you were performing, browser or device used, and whether other users are affected. Do not include passwords or sensitive customer data in the ticket.",
        "title": "Business app support"
    },
    {
        "question": "Can I paste customer personal data into an IT ticket?",
        "answer": "Only include the minimum information needed for troubleshooting and mask sensitive values when possible. If full customer data is required, use the approved secure attachment method and mark the ticket with the correct data classification.",
        "title": "Customer data in tickets"
    },
    {
        "question": "What does remote wipe do on a managed phone?",
        "answer": "Remote wipe removes company data and managed apps from the device after a loss, theft, offboarding, or security event. On personal devices, IT uses selective wipe where available to remove only company-managed data.",
        "title": "Remote wipe"
    },
    {
        "question": "How long will it take IT to respond to my ticket?",
        "answer": "Response time depends on priority. Critical outages are handled immediately, high-impact issues are prioritized the same business day, and standard requests are handled according to the service catalog SLA.",
        "title": "SLA response time"
    }
]

_MODEL = None
_TOKENIZER = None


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _fallback_answer(question: str) -> str:
    q_tokens = _tokens(question)
    best = None
    best_score = -1
    for policy in POLICIES:
        score = len(q_tokens & (_tokens(policy["question"]) | _tokens(policy["title"])))
        if score > best_score:
            best = policy
            best_score = score
    if best and best_score > 0:
        return best["answer"]
    return (
        "Please open an IT helpdesk ticket with the application or device name, exact error, "
        "screenshot, timestamp, business impact, and steps already tried. Do not share passwords "
        "or confidential data in chat."
    )


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER
    if not (FINAL_MODEL_DIR / "adapter_config.json").exists():
        return None, None
    try:
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(FINAL_MODEL_DIR),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        _MODEL, _TOKENIZER = model, tokenizer
        return model, tokenizer
    except Exception as exc:
        print(f"Model load failed, using retrieval fallback: {exc}")
        return None, None


def generate_answer(question: str) -> str:
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        return _fallback_answer(question)

    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([text], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.2,
        do_sample=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split(question)[-1].strip() or decoded


if __name__ == "__main__":
    print("HelpDeskAI ready. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        print("\nAnswer:", generate_answer(question))
