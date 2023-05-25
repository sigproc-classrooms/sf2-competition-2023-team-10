# SF2 Competition entry

`sf2_competition_template` is a template repository for your competition entry.
Your repository must provide the three functions within the `competition` module:

* `header_bits`
* `encode`
* `decode`

To help you get started, we've preloaded this template with an encoder that uses `jpegenc` and
`jpegdec`, along with the `opthuff` argument to build the header.

Note that this respository does not include a copy of [`cued_sf2_lab`](https://github.com/sigproc/cued_sf2_lab), but will find the version that you already have installed.
As such, you **should not modify your `cued_sf2_lab`**, as the demonstrators will not have this version installed.
If you need to make changes to the function in that package, you should copy the files into this repository.

For development, you can import these functions in Jupyter with
```python
from competition import header_bits, encode, decode
```

In the competition, we will call these functions with the appropriate inputs and outputs to generate the codewords, header, and decoded image.
When you push your code to github, this will be done automatically by https://github.com/sigproc-classrooms/sf2_competition_action!
To see these results:
* Click the green checkmark (or red cross) next to your commit message
* Click "details" on the SF2 Competition job
* Click the "🏠 summary" button on the top left
* Scroll down to see the report

The readme in https://github.com/sigproc-classrooms/sf2_competition_action may be updated with more info during the competition.

If you get a red cross and can't work out why, ask a demonstrator! This tool is experimental for 2022.
