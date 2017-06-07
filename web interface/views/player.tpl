% rebase('layout.tpl', title='Start', year=year)

<div class="jumbotron">
    <h1>Music genre classification.</h1>
    <p class="lead">Welcome to music genre classification, site for experts. Here you can listen to some music and predict its genre.</p>
    <p><a href="https://github.com/morcae/classification" class="btn btn-primary btn-large">Github &raquo;</a></p>
</div>
	<center>
		<audio controls>
			  <source src={{!path_audio}} type="audio/wav">
					Your browser does not support the audio element.
		</audio>
        <form name="options" method="POST" action="/second">
			<div class="checkboxgroup">
				  <input type="radio" name="genre" value={{!first}} > {{!first}} </input><br>
			</div>
			<div class="checkboxgroup">
				  <input type="radio" name="genre" value={{!second}} > {{!second}} </input><br>
			</div>
			<div class="checkboxgroup">
				  <input type="radio" name="genre" value={{ !third}} > {{ !third}} </input><br>
			</div>

			<div class="checkboxgroup">
				  <input name="Start" type="submit" value="Submit results">
			</div>

        </form>
	</center>
</div>
