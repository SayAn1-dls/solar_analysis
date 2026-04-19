with open("app.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_tab3_eval = False
in_tab5_exp = False

for i, line in enumerate(lines):
    # Tab 3 start of evaluation
    if 'preds = np.clip(model.predict(X_test_ts), 0, None)' in line and 'preds_export' not in line:
        line = line.replace('model.predict', 'load_model().predict')
        new_lines.append('    if st.button("Run Model Evaluation", key="eval_btn"):\n')
        new_lines.append('        st.session_state["show_eval"] = True\n\n')
        new_lines.append('    if st.session_state.get("show_eval", False):\n')
        new_lines.append('    ' + line)
        in_tab3_eval = True
        continue
        
    if in_tab3_eval:
        if line.startswith('with tab4:'):
            in_tab3_eval = False
            new_lines.append(line)
            continue
            
        if line.strip() == "":
            new_lines.append(line)
        elif line.startswith('    '):
            line = line.replace("model.feature_importances_", "load_model().feature_importances_")
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
        continue

    # Tab 5 export
    if 'preds_export = np.clip(model.predict(X_test_ts), 0, None)' in line:
        line = line.replace('model.predict', 'load_model().predict')
        new_lines.append('    if st.button("Generate Test Set Export", key="export_btn"):\n')
        new_lines.append('        st.session_state["show_export"] = True\n\n')
        new_lines.append('    if st.session_state.get("show_export", False):\n')
        new_lines.append('    ' + line)
        in_tab5_exp = True
        continue
        
    if in_tab5_exp:
        if line.startswith('with tab6:'):
            in_tab5_exp = False
            new_lines.append(line)
            continue
            
        if line.strip() == "":
            new_lines.append(line)
        elif line.startswith('    '):
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
        continue

    new_lines.append(line)

with open("app.py", "w") as f:
    f.writelines(new_lines)
