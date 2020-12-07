
/*  Extension to meramid.js to re-rerender Mermaids

  v0.1
  https://github.com/s0ubhik/jekyll-theme-chirpy
  © 2020 Soubhik Biswas
  MIT License
*/
class _mermaid {
	constructor(Class){
		if (Class == undefined) Class = "mermaid"
		this.graphs = []
		let mrmds  = document.getElementsByClassName(Class)
		this.mrmds = mrmds
        let graps = []
        for (let i=0; i < mrmds.length; i++){
	      let mrmd = mrmds[i]
	      let ID = "Mermaid-"+i
	      let graph = mrmd.innerText
	      mrmd.setAttribute("id",ID)
	      this.graphs.push({id: ID, graph: graph})
        }
	}

	async clean(){
		for (let i=0; i < this.mrmds.length; i++){
			if (this.mrmds[i].getAttribute("id") == this.graphs[i].id){
				this.mrmds[i].innerText = this.graphs[i].graph
				this.mrmds[i].removeAttribute("data-processed")
			}
		}
	}

	async render(init){
		await this.clean()
		if (init == null) init = {}
		mermaid.initialize(init)
		mermaid.init(init, this.mrmds)
		
	}
}
