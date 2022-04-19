import re

dest_dir = "docs/_static"
src_dir = "docs/cdocs"

srcs = ["craytraverse"]

for src in srcs:
    f = open(f"{src_dir}/_build/html/{src}.html", 'r')
    htmltxt = f.read()
    f.close()
    htmltxt = htmltxt.split('<div class="body" role="main">')[-1]
    htmltxt = htmltxt.rsplit('</section>', 1)[0] + '</section>'
    # htmltxt = re.split(r'(</dd>\s*</dl>\s*</div>)', htmltxt)
    f = open(f"{dest_dir}/{src}.html", 'w')
    f.write(htmltxt)
    # f.write(''.join(htmltxt[:-1]))
    f.close()
