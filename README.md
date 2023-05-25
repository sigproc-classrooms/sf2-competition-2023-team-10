# SF2 Competition entry

`sf2_competition_template` is a template repository for your competition entry.
You might be reading this readme in your own assignment repository (`sf2-competition-YYYY-team-N` or similar); if so, make sure to check out the [Template readme](https://github.com/sigproc-classrooms/sf2_competition_template/blob/main/README.md) for any updated advice.

---

Your repository must provide the three functions within the `competition` module:

* `header_bits`
* `encode`
* `decode`

To help you get started, we've preloaded this template with an encoder that uses `jpegenc` and
`jpegdec`, along with the `opthuff` argument to build the header.

Note that this respository does not include a copy of [`cued_sf2_lab`](https://github.com/sigproc/cued_sf2_lab), but will find the version that you already have installed.
As such, you **should not modify your `cued_sf2_lab`**, as the demonstrators will not have this version installed.
If you need to make changes to a function in that package, you should copy the modified functions into this repository.

For development, you can import these functions in Jupyter with
```python
from competition import header_bits, encode, decode
```

In the competition, we will call these functions with the appropriate inputs and outputs to generate the codewords, header, and decoded image.
When you push your code to github, this will be done automatically by https://github.com/sigproc-classrooms/sf2_competition_action!

### How do I see my automated build?

To see these results:
* Click the green checkmark (or red cross) next to your commit message
* Click "details" on the SF2 Competition job
* Click the "🏠 summary" button on the top left
* Scroll down to see the report

### How should I test my code

The best way to test your code is just to push it and look at the output described above!

However, you can also use the `cued_sf2_compete` command line tool, which is [documented here](https://github.com/sigproc-classrooms/sf2_competition_action/blob/2023/README.md).

### How do I install this locally

If you don't want to use github codespaces, you can clone this repository and run `pip install -r requirements.txt`.

### How can I make `plt.show()` work in Github Codespaces?

If opening in codespaces, you'll see a 'Ports' tab in the bottom pane in vscode.
1. Switch to this tab
2. Find the row that says "Open GUI (6080)" or similar.
3. Click the 🌐 symbol that appears in the "Local address" column; or right click on the row and click "Open in browser"
4. When asked for a password, use `vscode`.

Note that students of the 2022 course will need to manually update their `.devcontainer/devcontainer.json` to get this feature, by copying it into their own repository.

### I have more questions

The readme in https://github.com/sigproc-classrooms/sf2_competition_action may be updated with more info during the competition.

If you get a red cross and can't work out why, ask a demonstrator! This tool is experimental for 2022.
