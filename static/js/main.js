document.addEventListener("DOMContentLoaded", () => {
    
    // --- Clock ---
    setInterval(() => {
        const now = new Date();
        const str = now.getFullYear() + "-" + 
                    String(now.getMonth()+1).padStart(2,'0') + "-" + 
                    String(now.getDate()).padStart(2,'0') + "  " + 
                    String(now.getHours()).padStart(2,'0') + ":" + 
                    String(now.getMinutes()).padStart(2,'0') + ":" + 
                    String(now.getSeconds()).padStart(2,'0');
        document.getElementById("clock").innerText = str;
    }, 1000);

    // --- State Management ---
    const updateConfig = async (data) => {
        await fetch('/api/config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
    };

    // UI Elements
    const sldThr = document.getElementById("slider-thr");
    const valThr = document.getElementById("val-thr");
    const spinVote = document.getElementById("spin-vote");
    const backendSel = document.getElementById("backend-sel");
    const deepfaceSel = document.getElementById("deepface-sel");

    sldThr.oninput = () => {
        const val = (sldThr.value / 100).toFixed(2);
        valThr.innerText = val;
        updateConfig({threshold: parseFloat(val)});
    };

    spinVote.onchange = () => {
        updateConfig({vote_frames: parseInt(spinVote.value)});
    };

    backendSel.onchange = () => {
        if(backendSel.value === "deepface") deepfaceSel.style.display = "inline-block";
        else deepfaceSel.style.display = "none";
        updateConfig({backend: backendSel.value});
    };

    deepfaceSel.onchange = () => {
        updateConfig({deepface_model: deepfaceSel.value});
    };

    // --- Workflow Handlers ---
    
    // Enroll
    document.getElementById("btn-enroll").onclick = async () => {
        const name = document.getElementById("enroll-name").value;
        const count = document.getElementById("enroll-count").value;
        if(!name.trim()) return alert("Please enter a subject name.");
        
        document.getElementById("btn-enroll").disabled = true;
        document.getElementById("btn-enroll").innerText = "● CAPTURING...";
        document.getElementById("feed-status").innerText = "● ENROLL";
        document.getElementById("feed-status").style.color = "var(--cyan)";

        await fetch('/api/enroll', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name, count: parseInt(count)})
        });
    };

    // Train
    document.getElementById("btn-train").onclick = async () => {
        document.getElementById("btn-train").disabled = true;
        document.getElementById("btn-train").innerText = "⬡ PROCESSING...";
        backendSel.disabled = true;
        deepfaceSel.disabled = true;

        await fetch('/api/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                backend: backendSel.value,
                deepface_model: deepfaceSel.value
            })
        });
    };

    // Evaluate
    document.getElementById("btn-eval").onclick = async () => {
        document.getElementById("btn-eval").disabled = true;
        document.getElementById("btn-eval").innerText = "◫ EVALUATING...";
        
        const res = await fetch('/api/evaluate', {method: 'POST'});
        const data = await res.json();
        
        document.getElementById("btn-eval").disabled = false;
        document.getElementById("btn-eval").innerText = "◫ EVALUATE ACCURACY";
        
        if(data.error) {
            alert(data.error);
        } else {
            document.getElementById("eval-report").innerText = data.report;
            document.getElementById("eval-modal").style.display = "block";
        }
    };

    document.querySelector(".close-btn").onclick = () => {
        document.getElementById("eval-modal").style.display = "none";
    };

    // Monitor Toggle
    document.getElementById("btn-monitor-start").onclick = async () => {
        await fetch('/api/monitor/toggle', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'start'})
        });
        document.getElementById("btn-monitor-start").style.display = "none";
        document.getElementById("btn-monitor-stop").style.display = "block";
    };

    document.getElementById("btn-monitor-stop").onclick = async () => {
        await fetch('/api/monitor/toggle', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'stop'})
        });
        document.getElementById("btn-monitor-start").style.display = "block";
        document.getElementById("btn-monitor-stop").style.display = "none";
    };

    // --- State Polling ---
    setInterval(async () => {
        try {
            const res = await fetch('/api/state');
            const data = await res.json();
            
            // Enroll reset
            if(data.mode !== 'enroll' && document.getElementById("btn-enroll").disabled) {
                document.getElementById("btn-enroll").disabled = false;
                document.getElementById("btn-enroll").innerText = "◉ CAPTURE FACE";
                document.getElementById("enroll-name").value = "";
                document.getElementById("feed-status").innerText = "● OFFLINE";
                document.getElementById("feed-status").style.color = "var(--text-dim)";
                refreshUsers();
                if(data.enroll_msg) {
                    document.getElementById("enroll-msg").innerText = data.enroll_msg;
                    setTimeout(() => document.getElementById("enroll-msg").innerText = "", 5000);
                }
            }
            
            // Train Process
            document.getElementById("train-pct").innerText = data.train_progress + "%";
            document.getElementById("train-fill").style.width = data.train_progress + "%";
            document.getElementById("train-status").innerText = data.train_status;
            
            if(data.train_status === "DONE" || data.train_status === "ERROR") {
                if(document.getElementById("btn-train").disabled) {
                    document.getElementById("btn-train").disabled = false;
                    document.getElementById("btn-train").innerText = "⬡ BUILD EMBEDDINGS";
                    backendSel.disabled = false;
                    deepfaceSel.disabled = false;
                    if(data.train_message) alert(data.train_message);
                }
            }

            // Monitor Error Catch
            if(data.monitor_error) {
                alert(data.monitor_error);
                document.getElementById("btn-monitor-start").style.display = "block";
                document.getElementById("btn-monitor-stop").style.display = "none";
            }
            
            // Mode Pills
            document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
            if(data.mode === 'enroll') document.getElementById("pill-camera").classList.add("active");
            else if(data.mode === 'train') document.getElementById("pill-embeddings").classList.add("active");
            else if(data.mode === 'monitor') document.getElementById("pill-model").classList.add("active");
            else document.getElementById("pill-system").classList.add("active");

            // Feed
            if(data.mode === 'monitor') {
                document.getElementById("feed-status").innerText = "● MONITORING";
                document.getElementById("feed-status").style.color = "var(--green)";
            }
            
            // Latest Detection
            const detEl = document.getElementById("latest-detection");
            detEl.innerText = data.monitor_last_detection || "— AWAITING STREAM";
            if(data.monitor_last_detection !== "— AWAITING STREAM" && data.mode === 'monitor') {
                detEl.style.color = "var(--green)";
            } else {
                detEl.style.color = "var(--text-dim)";
            }

        } catch(e) {}
    }, 1000);

    // --- Users ---
    const refreshUsers = async () => {
        const res = await fetch('/api/users');
        const data = await res.json();
        const list = document.getElementById("users-list");
        list.innerHTML = "";
        data.users.forEach(u => {
            const sizeText = u.size_human ? ` | ${u.size_human}` : "";
            list.innerHTML += `<div class="user-item"><span class="u-name">${u.name}</span><span class="u-count">${u.count} imgs${sizeText}</span></div>`;
        });
    };
    
    document.getElementById("btn-refresh-users").onclick = refreshUsers;
    refreshUsers();
});
