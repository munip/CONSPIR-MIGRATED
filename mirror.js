// Mirrors the Station House platform's project overview and task board, live.
// Data comes from a public read-only endpoint; everything is rendered with
// textContent, so nothing from the response is ever parsed as HTML.
(function () {
	var DEFAULT_API = "https://app.station-house.net/api/public/conspiracybench";
	var root = document.getElementById("platform-mirror");
	if (!root) return;

	// ?api=... lets a maintainer preview against a local platform build.
	var override = new URLSearchParams(location.search).get("api");
	var api = override && /^https?:\/\//.test(override) ? override : DEFAULT_API;

	function el(tag, cls, text) {
		var node = document.createElement(tag);
		if (cls) node.className = cls;
		if (text != null) node.textContent = text;
		return node;
	}

	function link(url, text) {
		if (!/^https?:\/\//.test(url || "")) return el("span", null, text);
		var a = el("a", null, text);
		a.href = url;
		a.target = "_blank";
		a.rel = "noopener noreferrer";
		return a;
	}

	function daysLeft(deadline) {
		var today = new Date();
		today.setHours(0, 0, 0, 0);
		return Math.round((new Date(deadline + "T00:00:00") - today) / 86400000);
	}

	function taskCard(task, muted) {
		var card = el("div", "task" + (muted ? " muted" : ""));
		card.title = task.completionCriteria;
		card.appendChild(muted || !task.url ? el("span", null, task.title) : link(task.url, task.title));
		var tags = el("span", "tags");
		if (task.daily) tags.appendChild(el("span", "tag daily", "Daily"));
		if (task.deadline && !muted) {
			var d = daysLeft(task.deadline);
			if (d >= 0) tags.appendChild(el("span", "tag deadline", d === 0 ? "Last day" : d === 1 ? "1 day left" : d + " days left"));
		}
		if (tags.childNodes.length) card.appendChild(tags);
		return card;
	}

	function column(title, sections, muted) {
		var col = el("div", "column");
		col.appendChild(el("h4", null, title));
		var count = 0;
		sections.forEach(function (section) {
			if (section.title) col.appendChild(el("p", "group", section.title));
			section.tasks.forEach(function (task) {
				col.appendChild(taskCard(task, muted));
				count++;
			});
		});
		if (!count) col.appendChild(el("p", "empty", "Nothing here yet."));
		return col;
	}

	function render(data) {
		root.textContent = "";

		root.appendChild(el("h2", null, "Project status"));
		var phases = el("div", "phases");
		data.summary.phases.forEach(function (phase) {
			var box = el("div", "phase" + (phase.current ? " current" : ""));
			var head = el("p", "phase-title", phase.title);
			if (phase.current) head.appendChild(el("span", "tag now", "Now"));
			box.appendChild(head);
			box.appendChild(el("p", null, phase.description));
			phases.appendChild(box);
		});
		root.appendChild(phases);

		var timeline = el("ol", "timeline");
		data.milestones.forEach(function (m) {
			var li = el("li", m.status, m.title);
			if (m.status === "current") li.appendChild(el("span", "tag now", "In progress"));
			timeline.appendChild(li);
		});
		root.appendChild(timeline);

		root.appendChild(el("h2", null, "Task board"));
		var intro = el("p", null, "Open tasks for contributors. Sign in on the platform to start one and track your progress: ");
		intro.appendChild(link(data.platformUrl, "app.station-house.net"));
		root.appendChild(intro);
		var board = el("div", "board");
		board.appendChild(column("Current tasks", data.board.current, false));
		if (data.board.previous.some(function (s) { return s.tasks.length; })) {
			board.appendChild(column("Previous tasks", data.board.previous, true));
		}
		board.appendChild(column("Future tasks", data.board.future, true));
		root.appendChild(board);

		root.appendChild(el("h2", null, "Contributions leaderboard"));
		if (!data.leaderboard.length) {
			root.appendChild(el("p", null, "No contributions yet."));
		} else {
			var table = el("table", "leaderboard");
			var head = el("tr");
			["#", "Contributor", "Merged", "Open", "Discussions & reviews"].forEach(function (h) { head.appendChild(el("th", null, h)); });
			table.appendChild(head);
			data.leaderboard.forEach(function (row, i) {
				var tr = el("tr");
				[i + 1, row.author, row.merged, row.open, row.involvements].forEach(function (v) { tr.appendChild(el("td", null, String(v))); });
				table.appendChild(tr);
			});
			root.appendChild(table);
		}

		if (data.bibliography.length) {
			root.appendChild(el("h2", null, "Bibliography"));
			var bib = el("ul");
			data.bibliography.forEach(function (b) {
				var li = el("li");
				li.appendChild(link(b.url, b.title));
				li.appendChild(el("span", "muted-text", " — " + b.takeaway));
				bib.appendChild(li);
			});
			root.appendChild(bib);
		}

		if (data.educationalLinks.length) {
			root.appendChild(el("h2", null, "Educational links"));
			[["intro", "Start here"], ["advanced", "Advanced / research"]].forEach(function (pair) {
				var items = data.educationalLinks.filter(function (l) { return l.section === pair[0]; });
				if (!items.length) return;
				root.appendChild(el("h3", null, pair[1]));
				var ul = el("ul");
				items.forEach(function (l) {
					var li = el("li");
					li.appendChild(link(l.url, l.title));
					var meta = [l.source, l.length].filter(Boolean).join(" · ");
					if (meta) li.appendChild(el("span", "muted-text", " (" + meta + ")"));
					ul.appendChild(li);
				});
				root.appendChild(ul);
			});
		}

		root.appendChild(el("p", "updated", "Live from the Station House platform · updated " + new Date(data.generatedAt).toLocaleString()));
	}

	fetch(api)
		.then(function (res) {
			if (!res.ok) throw new Error("HTTP " + res.status);
			return res.json();
		})
		.then(render)
		.catch(function () {
			root.textContent = "";
			var p = el("p", null, "Couldn't load the live project status. See it on ");
			p.appendChild(link("https://app.station-house.net/projects/conspiracybench", "app.station-house.net"));
			root.appendChild(p);
		});
})();
