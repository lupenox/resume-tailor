import re

with open("resume_tailor/static/app.css", "r") as f:
    css = f.read()

# Extract Analytics
analytics_match = re.search(r'/\* --- Analytics Page Specifics ---\s*\*/.*?(?=/\* ---|$)', css, re.DOTALL)
analytics_css = analytics_match.group(0) if analytics_match else ""
css = css.replace(analytics_css, "")

# Extract History
history_match = re.search(r'/\* --- History List \(Scrollable\) ---\s*\*/.*?(?=/\* ---|/\* Status Pills|$)', css, re.DOTALL)
history_css = history_match.group(0) if history_match else ""
css = css.replace(history_css, "")

# Extract Master Resume Actions
master_match = re.search(r'/\* --- Master Resume Actions ---\s*\*/.*?(?=/\* ---|$)', css, re.DOTALL)
master_css = master_match.group(0) if master_match else ""
css = css.replace(master_css, "")

# Extract Empty State
empty_match = re.search(r'/\* Empty State \*/.*?(?=/\* ---|/\* Analytics|$)', css, re.DOTALL)
empty_css = empty_match.group(0) if empty_match else ""
css = css.replace(empty_css, "")

# Extract Choice Grid & Cards
choice_match = re.search(r'/\* Choice Grid & Cards \*/.*?(?=/\* Segmented|$)', css, re.DOTALL)
choice_css = choice_match.group(0) if choice_match else ""
css = css.replace(choice_css, "")

dashboard_css = "\n\n".join(filter(None, [choice_css, history_css, empty_css, master_css]))

with open("resume_tailor/static/analytics.css", "w") as f:
    f.write(analytics_css)

with open("resume_tailor/static/dashboard.css", "w") as f:
    f.write(dashboard_css)

with open("resume_tailor/static/app.css", "w") as f:
    f.write(css)

print("CSS refactored!")
