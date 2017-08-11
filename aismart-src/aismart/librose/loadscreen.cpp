/* $Id: loadscreen.cpp 47608 2010-11-21 01:56:29Z shadowmaster $ */
/*
   Copyright (C) 2005 - 2010 by Joeri Melis <joeri_melis@hotmail.com>
   Part of the Battle for Wesnoth Project http://www.wesnoth.org/

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY.

   See the COPYING file for more details.
*/

/**
 * @file
 * Screen with logo and "Loading ..."-progressbar during program-startup.
 */
#define GETTEXT_DOMAIN "rose-lib"

#include "loadscreen.hpp"

#include "log.hpp"
#include "font.hpp"
#include "marked-up_text.hpp"
#include "gettext.hpp"
#include "filesystem.hpp"
#include "video.hpp"
#include "image.hpp"
#include "wml_exception.hpp"
#include "rose_config.hpp"

#include <SDL_events.h>
#include <SDL_image.h>

#include <cassert>

static lg::log_domain log_display("display");
static lg::log_domain log_loadscreen("loadscreen");
#define LOG_LS LOG_STREAM(info, log_loadscreen)
#define ERR_DP LOG_STREAM(err, log_display)

loadscreen::global_loadscreen_manager* loadscreen::global_loadscreen_manager::manager = 0;

loadscreen::global_loadscreen_manager::global_loadscreen_manager(CVideo& screen)
  : owns(global_loadscreen == 0)
{
	if(owns) {
		manager = this;
		global_loadscreen = new loadscreen(screen);
		global_loadscreen->clear_screen();
	}
}

loadscreen::global_loadscreen_manager::~global_loadscreen_manager()
{
	reset();
}

void loadscreen::global_loadscreen_manager::reset()
{
	if(owns) {
		owns = false;
		manager = 0;
		assert(global_loadscreen);
		// global_loadscreen->clear_screen();
		delete global_loadscreen;
		global_loadscreen = 0;
	}
}

static surface scaled_logo_surface;

loadscreen::loadscreen(CVideo &screen, const int &percent):
	screen_(screen),
	textarea_(),
	logo_surface_(image::get_image(game_config::logo_png)),
	logo_drawn_(false),
	pby_offset_(0),
	prcnt_(percent)
{
	if (logo_surface_.null()) {
		ERR_DP << "loadscreen: Failed to load the logo" << std::endl;
	}
	textarea_.x = textarea_.y = textarea_.w = textarea_.h = 0;
}

void loadscreen::draw_screen(const std::string &text)
{
	return;
/*
	// Set progress bar parameters:
	//
	// RGB-values for finished piece.
	int fcr =  21, fcg =  53, fcb =  80;
	// Groove.
	int lcr =  21, lcg =  22, lcb =  24;
	// Border color.
	int bcr = 188, bcg = 176, bcb = 136;
	// Border width.
	int bw = 1;
	// Border inner spacing width.
	int bispw = 1;
	bw = 2*(bw+bispw) > screen_.getx() ? 0: 2*(bw+bispw) > screen_.gety() ? 0: bw;
	// Available width.
	int scrx = screen_.getx() - 2*(bw+bispw);
	// Available height.
	int scry = screen_.gety() - 2*(bw+bispw);
	// Used width.
	int pbw = scrx/2;
	// Used height.
	int pbh = scry/16;
	// Height of the lighting line.
	int	lightning_thickness = 2;

	surface gdis = screen_.getSurface();
	SDL_Rect area;

	// Pump events and make sure to redraw the logo if there's a chance that it's been obscured
	SDL_Event ev;
	while(SDL_PollEvent(&ev)) {
		if (ev.type == SDL_WINDOWEVENT) {
			if (ev.window.event == SDL_WINDOWEVENT_RESIZED) {
				logo_drawn_ = false;
			}
		}
	}

	if (game_config::app == "sleep" || game_config::app == "sesame" || game_config::app == "blesmart") {
		// Update the rectangle.
		if (!scaled_logo_surface) {
			scaled_logo_surface = scale_surface(logo_surface_, gdis->w, gdis->h);
		}
		area.x = area.y = 0;
		sdl_blit(scaled_logo_surface, 0, gdis, &area);
		screen_.flip();
		return;
	}

	// Draw logo if it was successfully loaded.
	if (logo_surface_ && !logo_drawn_) {
		area.x = (screen_.getx () - logo_surface_->w) / 2;
		area.y = ((scry - logo_surface_->h) / 2) - pbh;
		area.w = logo_surface_->w;
		area.h = logo_surface_->h;
		// Check if we have enough pixels to display it.
		if (area.x > 0 && area.y > 0) {
			pby_offset_ = (pbh + area.h)/2;
			sdl_blit(logo_surface_, 0, gdis, &area);
		} else {
			ERR_DP << "loadscreen: Logo image is too big." << std::endl;
		}
		logo_drawn_ = true;
	}
	int pbx = (scrx - pbw)/2;					// Horizontal location.
	int pby = (scry - pbh)/2 + pby_offset_;		// Vertical location.

	// Draw top border.
	area.x = pbx; area.y = pby;
	area.w = pbw + 2*(bw+bispw); area.h = bw;
	sdl_fill_rect(gdis,&area,SDL_MapRGB(gdis->format,bcr,bcg,bcb));
	// Draw bottom border.
	area.x = pbx; area.y = pby + pbh + bw + 2*bispw;
	area.w = pbw + 2*(bw+bispw); area.h = bw;
	sdl_fill_rect(gdis,&area,SDL_MapRGB(gdis->format,bcr,bcg,bcb));
	// Draw left border.
	area.x = pbx; area.y = pby + bw;
	area.w = bw; area.h = pbh + 2*bispw;
	sdl_fill_rect(gdis,&area,SDL_MapRGB(gdis->format,bcr,bcg,bcb));
	// Draw right border.
	area.x = pbx + pbw + bw + 2*bispw; area.y = pby + bw;
	area.w = bw; area.h = pbh + 2*bispw;
	sdl_fill_rect(gdis,&area,SDL_MapRGB(gdis->format,bcr,bcg,bcb));
	// Draw the finished bar area.
	area.x = pbx + bw + bispw; area.y = pby + bw + bispw;
	area.w = (prcnt_ * pbw) / 100; area.h = pbh;
	sdl_fill_rect(gdis,&area,SDL_MapRGB(gdis->format,fcr,fcg,fcb));

	SDL_Rect lightning = area;
	lightning.h = lightning_thickness;
	//we add 25% of white to the color of the bar to simulate a light effect
	sdl_fill_rect(gdis,&lightning,SDL_MapRGB(gdis->format,(fcr*3+255)/4,(fcg*3+255)/4,(fcb*3+255)/4));
	lightning.y = area.y+area.h-lightning.h;
	//remove 50% of color to simulate a shadow effect
	sdl_fill_rect(gdis,&lightning,SDL_MapRGB(gdis->format,fcr/2,fcg/2,fcb/2));

	// Draw the leftover bar area.
	area.x = pbx + bw + bispw + (prcnt_ * pbw) / 100; area.y = pby + bw + bispw;
	area.w = ((100 - prcnt_) * pbw) / 100; area.h = pbh;
	sdl_fill_rect(gdis,&area,SDL_MapRGB(gdis->format,lcr,lcg,lcb));

	// Clear the last text and draw new if text is provided.
	if (!text.empty())
	{
		SDL_Rect oldarea = textarea_;
		sdl_fill_rect(gdis,&textarea_,SDL_MapRGB(gdis->format,0,0,0));
		textarea_ = font::line_size(text, font::SIZE_DEFAULT);
		textarea_.x = scrx/2 + bw + bispw - textarea_.w / 2;
		textarea_.y = pby + pbh + 4*(bw + bispw);

		surface surf = font::get_rendered_text(text, textarea_.w, font::SIZE_DEFAULT, font::NORMAL_COLOR);
		textarea_.w = surf->w;
		textarea_.h = surf->h;
		SDL_Rect dst = textarea_;
		sdl_blit(surf, NULL, screen_.getSurface(), &dst);

		SDL_Rect refresh = union_rects(oldarea, textarea_);
	}

	// Update the rectangle.
	screen_.flip();
*/
}

void loadscreen::clear_screen()
{
	return;
/*
	int scrx = screen_.getx();                     //< Screen width.
	int scry = screen_.gety();                     //< Screen height.
	SDL_Rect area = create_rect(0, 0, scrx, scry); // Screen area.
	surface disp(screen_.getSurface());      // Screen surface.
	// Make everything black.
	sdl_fill_rect(disp,&area,SDL_MapRGB(disp->format,0,0,0));
	screen_.flip();
*/
}

loadscreen *loadscreen::global_loadscreen = 0;

struct load_stage
{
	char const *id;
	char const *name;
	int start_pos, max_count;
};

static int const nb_stages = 20;

/**
 * Description of all the stages.
 * @note Some of the stages appear twice; this is not a mistake. It
 *       accounts for their cost at title time and at game time.
 * @note The values have been automatically generated by running a cache-hot
 *       Wesnoth on HttH with the --log-info=loadscreen option.
 */
static load_stage const stages[nb_stages] =
{
	{ "init gui", "Initializing user interface", 0, 17210 },
	{ "load config", "Loading game configuration", 20, 0 },
	{ "verify cache", "Verifying cache", 20, 164 },
	{ "create cache", "Reading files and creating cache", 21, 80287 },
	{ "load unit types", "Reading unit files", 94, 467 },
	{ "init fonts", "Re-initialize fonts for the current language", 98, 17 },
	{ "refresh addons", "Searching for installed add-ons", 99, 0 },
	{ "titlescreen", "Loading title screen", 100, 0 },
	{ "load data", "Loading data files", 0, 0 },
	{ "verify cache", "Verifying cache", 0, 0 },
	{ "create cache", "Reading files and creating cache", 0, 152109 },
	{ "load unit types", "Reading unit files", 74, 488 },
	{ "load level", "Loading level", 75, 0 },
	{ "init teams", "Initializing teams", 75, 0 },
	{ "load units", "Loading units", 77, 0 },
	{ "init scene", "Initializing display", 77, 0 },
	{ "build terrain", "Building terrain rules", 77, 1240 },
	{ "init display", "Initializing display", 90, 0 },
	{ "draw scene", "Drawing scene", 95, 0 },
	{ "start game", "Starting game", 100, 0 },
};

static int current_stage;
static int stage_counter[nb_stages];
static unsigned stage_time[nb_stages];

void loadscreen::start_stage(char const *id)
{
	assert(global_loadscreen);

	int s = -1;
	for (int i = 0; i < nb_stages; ++i) {
		int j = (i + current_stage) % nb_stages;
		if (strcmp(id, stages[j].id) == 0) {
			s = j;
			break;
		}
	}

	std::stringstream str;
	str << "unknown stage id: " << id;
	VALIDATE(s >= 0, str.str());

	const load_stage &cs = stages[s];
	global_loadscreen->prcnt_ = cs.start_pos;
	global_loadscreen->draw_screen(_(cs.name));
	stage_counter[s] = 0;
	stage_time[s] = SDL_GetTicks();
	current_stage = s;
}

void loadscreen::increment_progress()
{
	if (!global_loadscreen) return;

	int v = ++stage_counter[current_stage];
	int m = stages[current_stage].max_count;
	if (v > m) return;

	int s = stages[current_stage].start_pos;
	int percentage = s + v * (stages[current_stage + 1].start_pos - s) / m;
	if (percentage == global_loadscreen->prcnt_) return;

	global_loadscreen->prcnt_ = percentage;
	global_loadscreen->draw_screen(std::string());
}

void loadscreen::dump_counters() const
{
	if (lg::info.dont_log(log_loadscreen)) return;

	std::ostringstream s;

	int i = 0;
	while (i < nb_stages)
	{
		int j;
		for (j = i; stages[j].start_pos < 100; ++j) {}
		if (stage_time[i] == stage_time[j]) break;
		for (int k = i; k <= j; ++k)
		{
			int v = stages[k].start_pos;
			if (i < k && k < j) {
				v = stages[i].start_pos +
					(100 - stages[i].start_pos) *
					(stage_time[k] - stage_time[i]) /
					(stage_time[j] - stage_time[i]);
			}
			s << "\t{ \"" << stages[k].id << "\", N_(\""
			  << stages[k].name << "\"), " << v << ", "
			  << stage_counter[k] << " },\n";
		}
		i = j + 1;
	}
	LOG_LS << "Suggested loadscreen values:\n---\n" << s.str() << "---\n";
}


set_increment_progress::fn_increment_progress set_increment_progress::increment_progress = NULL;
void* set_increment_progress::ctx = NULL;

set_increment_progress::set_increment_progress(fn_increment_progress fn, void* ctx) :
	old_(increment_progress)
{
	set_increment_progress::increment_progress = fn;
	set_increment_progress::ctx = ctx;
}

set_increment_progress::~set_increment_progress()
{
	set_increment_progress::increment_progress = old_;
}

void increment_preprocessor_progress(std::string const &name, bool is_file)
{
	if (set_increment_progress::increment_progress) {
		set_increment_progress::increment_progress(name, is_file, set_increment_progress::ctx);
	}
}
